import CoreML
import Foundation

/// 各 CLI 子命令实现。所有对比同时输出可读摘要 + 可选 --out JSON。
enum Commands {

    // MARK: sched-check-fixture（无需锚点/模型）

    static func schedCheckFixture(out: String?) throws {
        let sched = DPMSolver()
        sched.setTimesteps(30)
        let ref = try jsonLoad(Paths.fixtures + "/schedule_ref.json")
        let refTs = (ref["timesteps"] as! [Any]).map { ($0 as! NSNumber).intValue }
        let refSig = (ref["sigmas"] as! [Any]).map { ($0 as! NSNumber).doubleValue }
        let refInit = (ref["init_noise_sigma"] as! NSNumber).doubleValue

        let tsOK = sched.timesteps == refTs
        var maxRelSig = 0.0
        for (a, b) in zip(sched.sigmas, refSig) {
            let denom = max(abs(b), 1e-30)
            maxRelSig = max(maxRelSig, abs(a - b) / denom)
        }
        // 末位 sigma=0 用绝对误差
        let lastAbs = abs(sched.sigmas.last! - refSig.last!)
        let sigOK = maxRelSig <= 1e-6 && lastAbs <= 1e-12
        let initOK = sched.initNoiseSigma == refInit
        // scale_model_input 恒等性
        let probe: [Float] = [1.5, -2.25, 0.125]
        let scaleOK = sched.scaleModelInput(probe, timestep: refTs[0]) == probe

        print("== schedule_ref ==")
        print("timesteps match: \(tsOK)  \(sched.timesteps)")
        print("sigmas max_rel_err: \(String(format: "%.3e", maxRelSig)) (tol 1e-6), last abs: \(lastAbs)")
        print("init_noise_sigma: \(sched.initNoiseSigma) (ref \(refInit))  scale identity: \(scaleOK)")

        // steps_ref：确定性逐步回放（显式 variance_noise，走完整 sde-dpmsolver++ 路径）
        let sr = try jsonLoad(Paths.fixtures + "/steps_ref.json")
        let n = (sr["shape"] as! [Any]).map { ($0 as! NSNumber).intValue }.reduce(1, *)
        var sample = (sr["sample0"] as! [Any]).map { Float(($0 as! NSNumber).doubleValue) }
        let mos = sr["model_outputs"] as! [Any]
        let vns = sr["variance_noises"] as! [Any]
        let refs = sr["latents_next"] as! [Any]
        let ts = (sr["timesteps"] as! [Any]).map { ($0 as! NSNumber).intValue }
        precondition(mos.count == 30 && n > 0)

        let sched2 = DPMSolver()
        sched2.setTimesteps(30)
        var rows: [[String: Any]] = []
        var maxAbsAll = 0.0, minPsnr = Double.infinity
        print("== steps_ref replay (30 steps) ==")
        for i in 0 ..< 30 {
            let mo = (mos[i] as! [Any]).map { Float(($0 as! NSNumber).doubleValue) }
            let vn = (vns[i] as! [Any]).map { Float(($0 as! NSNumber).doubleValue) }
            let next = sched2.step(modelOutput: mo, timestep: ts[i], sample: sample, varianceNoise: vn)
            let refNext = (refs[i] as! [Any]).map { Float(($0 as! NSNumber).doubleValue) }
            let (psnr, maxAbs) = dynPSNR(ref: refNext, out: next)
            maxAbsAll = max(maxAbsAll, maxAbs); minPsnr = min(minPsnr, psnr)
            rows.append(["step": i, "order": sched2.lastOrder, "psnr_db": psnr, "max_abs": maxAbs])
            if i < 3 || i >= 28 || maxAbs > 1e-3 {
                print(String(format: "  step %02d order=%d psnr=%@ dB max_abs=%.3e", i, sched2.lastOrder, fmt(psnr), maxAbs))
            }
            sample = next
        }
        let replayOK = maxAbsAll < 1e-3
        print(String(format: "replay summary: max_abs=%.3e min_psnr=%@ dB → %@", maxAbsAll, fmt(minPsnr), replayOK ? "PASS" : "FAIL"))
        let pass = tsOK && sigOK && initOK && scaleOK && replayOK
        jsonWrite([
            "command": "sched-check-fixture", "pass": pass,
            "timesteps_match": tsOK, "sigmas_max_rel_err": maxRelSig,
            "init_noise_sigma": sched.initNoiseSigma, "scale_identity": scaleOK,
            "replay_max_abs": maxAbsAll, "replay_min_psnr_db": minPsnr, "steps": rows,
        ], out)
        print(pass ? "PASS" : "FAIL")
        if !pass { exit(1) }
    }

    // MARK: sched-check --traj <dir>（锚点逐步对齐）

    static func schedCheck(traj: String, out: String?) throws {
        let meta = try TrajMeta.load(traj)
        try requireFiles([traj + "/latent_init.npy", traj + "/step_00_unet_out.npy", traj + "/step_00_latents_next.npy"],
                         hint: "golden_trajectory dump 未完成 → 待锚点")
        let n = meta.frames * 4 * 160 * 160
        let zeros = [Float](repeating: 0, count: n)
        let sched = DPMSolver()
        sched.setTimesteps(30)

        // pass 1: 零噪声确定性部分 vs 锚点（差 = K*noise，SDE 项）
        //         + 反推噪声统计（应≈N(0,1)，验证方程闭合）
        // pass 2: 灌入反推噪声 → 应与锚点逐位接近（验证噪声施加机制）
        var rows: [[String: Any]] = []
        var noises: [[Float]] = []
        var maxAbsReplay = 0.0
        let sched2 = DPMSolver(); sched2.setTimesteps(30)
        print("== sched-check \(meta.passName) (F=\(meta.frames)) ==")
        for i in 0 ..< 30 {
            let sample: [Float] = try i == 0
                ? Npy.load(traj + "/latent_init.npy").data
                : Npy.load(traj + String(format: "/step_%02d_latents_next.npy", i - 1)).data
            let mo = try Npy.load(traj + String(format: "/step_%02d_unet_out.npy", i)).data
            let ref = try Npy.load(traj + String(format: "/step_%02d_latents_next.npy", i)).data

            let det = sched.step(modelOutput: mo, timestep: meta.timesteps[i], sample: sample, varianceNoise: zeros)
            let k = sched.lastNoiseScale
            let (psnrDet, maxAbsDet) = dynPSNR(ref: ref, out: det)

            var implied = [Float](repeating: 0, count: n)
            var mean = 0.0, m2 = 0.0
            if k > 1e-12 {
                let invK = Float(1.0 / k)
                for j in 0 ..< n { implied[j] = (ref[j] - det[j]) * invK }
                for v in implied { let d = Double(v); mean += d }
                mean /= Double(n)
                for v in implied { let d = Double(v) - mean; m2 += d * d }
                m2 /= Double(n)
            }
            noises.append(implied)

            // pass 2 同一步：灌入反推噪声
            let replay = sched2.step(modelOutput: mo, timestep: meta.timesteps[i], sample: sample, varianceNoise: implied)
            let (psnrR, maxAbsR) = dynPSNR(ref: ref, out: replay)
            maxAbsReplay = max(maxAbsReplay, maxAbsR)

            rows.append(["step": i, "order": sched.lastOrder, "noise_scale_K": k,
                         "det_max_abs": maxAbsDet, "det_psnr_db": psnrDet,
                         "implied_noise_mean": mean, "implied_noise_std": m2.squareRoot(),
                         "replay_max_abs": maxAbsR, "replay_psnr_db": psnrR])
            print(String(format: "  step %02d order=%d K=%.4f det: max_abs=%.4f psnr=%@ dB | implied noise μ=%.4f σ=%.4f | replay max_abs=%.2e",
                         i, sched.lastOrder, k, maxAbsDet, fmt(psnrDet), mean, m2.squareRoot(), maxAbsR))
        }
        // 汇总：det 差应为 K 量级（≈0.1~1.2，即每步 SDE 噪声贡献）；
        // implied σ≈1 说明确定性系数全对；replay≈0 说明噪声项施加正确。
        // K=0 的末步无法反推噪声，det 与锚点差为 bf16 算术量化（锚点经 .to(bf16)），属预期。
        let stds = rows.compactMap { $0["implied_noise_std"] as? Double }.filter { $0 > 0 }
        let stdOK = stds.allSatisfy { abs($0 - 1.0) < 0.05 }
        let replayMidOK = rows.dropLast().allSatisfy { ($0["replay_max_abs"] as? Double ?? 1) < 1e-3 }
        let finalDet = (rows.last?["det_max_abs"] as? Double) ?? 0
        let finalOK = finalDet < 0.05  // bf16 量级（~0.008 相对）
        let pass = stdOK && replayMidOK && finalOK
        print(String(format: "summary: implied-noise σ range [%.4f, %.4f] (期望≈1), 前29步 replay max_abs≤1e-3: %@, 末步 det max_abs=%.4f (bf16 量化, 阈值 0.05) → %@",
                     stds.min() ?? 0, stds.max() ?? 0, replayMidOK ? "OK" : "NG", finalDet, pass ? "PASS" : "CHECK"))
        print("注: det-only 与锚点的差 = 每步 SDE 噪声项 K*ε（torch MPS 全局 RNG 产生，Swift 不可复现），")
        print("    反推噪声 σ≈1 且 replay 精确闭合即证明确定性系数与噪声系数全部正确。")
        jsonWrite(["command": "sched-check", "pass_name": meta.passName, "pass": pass,
                   "implied_noise_std_ok": stdOK, "replay_max_abs": maxAbsReplay, "steps": rows], out)
        if !pass { exit(1) }
    }

    /// 从锚点反推每步 SDE 噪声（denoise/e2e 对齐模式用）
    static func deriveStepNoises(traj: String, meta: TrajMeta) throws -> [[Float]] {
        let n = meta.frames * 4 * 160 * 160
        let zeros = [Float](repeating: 0, count: n)
        let sched = DPMSolver(); sched.setTimesteps(30)
        var noises: [[Float]] = []
        for i in 0 ..< 30 {
            let sample: [Float] = try i == 0
                ? Npy.load(traj + "/latent_init.npy").data
                : Npy.load(traj + String(format: "/step_%02d_latents_next.npy", i - 1)).data
            let mo = try Npy.load(traj + String(format: "/step_%02d_unet_out.npy", i)).data
            let ref = try Npy.load(traj + String(format: "/step_%02d_latents_next.npy", i)).data
            let det = sched.step(modelOutput: mo, timestep: meta.timesteps[i], sample: sample, varianceNoise: zeros)
            let k = sched.lastNoiseScale
            var noise = [Float](repeating: 0, count: n)
            if k > 1e-12 {
                let invK = Float(1.0 / k)
                for j in 0 ..< n { noise[j] = (ref[j] - det[j]) * invK }
            }
            noises.append(noise)
        }
        return noises
    }

    // MARK: CoreML UNet 单步 / 全循环

    static func unetInputs(sample: [Float], sampleShape: [Int], timestep: Int, statics: Statics) throws -> [String: MLMultiArray] {
        [
            "sample": try MLMultiArray(filledWith: sample, shape: sampleShape),
            "timestep": try MLMultiArray(filledWith: [Float(timestep)], shape: [1]),
            "encoder_hidden_states": try MLMultiArray(filledWith: statics.encoderHiddenStates, shape: statics.ehsShape),
            "text_embeds": try MLMultiArray(filledWith: statics.textEmbeds, shape: statics.teShape),
            "time_ids": try MLMultiArray(filledWith: statics.timeIds, shape: statics.tiShape),
        ]
    }

    static func unetStep(traj: String, stepN: Int, out: String?) throws {
        let meta = try TrajMeta.load(traj)
        let pass = meta.passName
        let bundle = try loadUNetBundle(pass: pass, traj: traj)
        let unet = bundle.model
        let statics = bundle.cond
        let sampleNpy = try Npy.load(traj + String(format: "/step_%02d_unet_in.npy", stepN))
        let refNpy = try Npy.load(traj + String(format: "/step_%02d_unet_out.npy", stepN))
        precondition(statics.ehsShape[0] == meta.frames, "条件 F 与 meta 不一致")

        let inputs = try unetInputs(sample: sampleNpy.data, sampleShape: sampleNpy.shape,
                                    timestep: meta.timesteps[stepN], statics: statics)
        var pred = ([Float](), [Int]())
        var times: [Double] = []
        for r in 0 ..< 3 {
            let t0 = nowMs()
            pred = try unet.predict(inputs, output: "noise_pred")
            times.append(nowMs() - t0)
            if r == 0 { print("[unet-step] first forward \(fmt(times[0], 0))ms（含编译/缓存）") }
        }
        let (psnr, maxAbs) = dynPSNR(ref: refNpy.data, out: pred.0)
        print(String(format: "unet-step %@ step %02d: psnr=%@ dB max_abs=%.4e | forward ms: first=%.0f warm=%@",
                     pass, stepN, fmt(psnr), maxAbs, times[0],
                     times.dropFirst().map { fmt($0, 0) }.joined(separator: ",")))
        jsonWrite(["command": "unet-step", "pass_name": pass, "step": stepN, "model": bundle.flavor,
                   "psnr_db": psnr, "max_abs": maxAbs,
                   "forward_ms_first": times[0], "forward_ms_warm": Array(times.dropFirst()),
                   "pred_shape": pred.1], out)
    }

    /// 完整 30 步 CoreML 去噪循环（对齐模式：初始 latent、c_concat、每步 SDE 噪声均来自锚点）
    /// 返回 (最终 latents (1,F,4,160,160), 逐步对比行, 每步前向 ms)
    static func denoiseLoop(traj: String, meta: TrajMeta, cConcat: [Float],
                            unet: CoreMLModel, statics: Statics, verbose: Bool) throws
        -> (latents: [Float], rows: [[String: Any]], forwardMs: [Double]) {
        let f = meta.frames
        let frameSize = 4 * 160 * 160
        let cConcatExp = expandFrames(cConcat, frames: f)
        let noises = try deriveStepNoises(traj: traj, meta: meta)
        var latents = try Npy.load(traj + "/latent_init.npy").data
        let sched = DPMSolver(); sched.setTimesteps(30)
        var rows: [[String: Any]] = []
        var forwardMs: [Double] = []
        let sampleShape = [1, f, 8, 160, 160]
        for i in 0 ..< 30 {
            var input = [Float](repeating: 0, count: 2 * f * frameSize)
            for fr in 0 ..< f {
                input.replaceSubrange((fr * 8) * frameSize / 4 ..< (fr * 8 + 4) * frameSize / 4,
                                      with: latents[fr * frameSize ..< (fr + 1) * frameSize])
                input.replaceSubrange((fr * 8 + 4) * frameSize / 4 ..< (fr * 8 + 8) * frameSize / 4,
                                      with: cConcatExp[fr * frameSize ..< (fr + 1) * frameSize])
            }
            let inputs = try unetInputs(sample: input, sampleShape: sampleShape,
                                        timestep: meta.timesteps[i], statics: statics)
            let t0 = nowMs()
            let pred = try unet.predict(inputs, output: "noise_pred").0
            let ms = nowMs() - t0
            forwardMs.append(ms)
            latents = sched.step(modelOutput: pred, timestep: meta.timesteps[i],
                                 sample: latents, varianceNoise: noises[i])
            var row: [String: Any] = ["step": i, "forward_ms": ms]
            let refPath = traj + String(format: "/step_%02d_latents_next.npy", i)
            if FileManager.default.fileExists(atPath: refPath) {
                let ref = try Npy.load(refPath).data
                let (psnr, maxAbs) = dynPSNR(ref: ref, out: latents)
                row["psnr_db"] = psnr; row["max_abs"] = maxAbs
                if verbose {
                    print(String(format: "  step %02d: forward=%.0fms latents psnr=%@ dB max_abs=%.4e", i, ms, fmt(psnr), maxAbs))
                }
            } else if verbose {
                print(String(format: "  step %02d: forward=%.0fms", i, ms))
            }
            rows.append(row)
        }
        return (latents, rows, forwardMs)
    }

    static func denoise(traj: String, out: String?) throws {
        let meta = try TrajMeta.load(traj)
        try requireFiles([traj + "/latent_init.npy", traj + "/c_concat.npy"], hint: "golden_trajectory dump 未完成 → 待锚点")
        let bundle = try loadUNetBundle(pass: meta.passName, traj: traj)
        let unet = bundle.model
        let statics = bundle.cond
        let cConcat = try Npy.load(traj + "/c_concat.npy").data
        print("== denoise \(meta.passName) (F=\(meta.frames), 30 steps, CoreML \(bundle.flavor)) ==")
        let (latents, rows, forwardMs) = try denoiseLoop(traj: traj, meta: meta, cConcat: cConcat,
                                                         unet: unet, statics: statics, verbose: true)
        var result: [String: Any] = ["command": "denoise", "pass_name": meta.passName, "model": bundle.flavor,
                                     "steps": rows, "forward_ms_total": forwardMs.reduce(0, +)]
        // 最终 decode_in 对比
        let decodeRefPath = traj + "/decode_in.npy"
        if FileManager.default.fileExists(atPath: decodeRefPath) {
            let ref = try Npy.load(decodeRefPath).data
            let inv = Float(1.0 / meta.scalingFactor)
            let decodeIn = latents.map { $0 * inv }
            let (psnr, maxAbs) = dynPSNR(ref: ref, out: decodeIn)
            print(String(format: "decode_in: psnr=%@ dB max_abs=%.4e", fmt(psnr), maxAbs))
            result["decode_in_psnr_db"] = psnr; result["decode_in_max_abs"] = maxAbs
        }
        print(String(format: "CoreML forward total %.1fs (avg %.0fms/step)",
                     forwardMs.reduce(0, +) / 1000, forwardMs.reduce(0, +) / 30))
        jsonWrite(result, out)
    }

    // MARK: decode --traj <dir>

    /// decode_in 逐帧解码 → RGBA8 图（对标 TransparentVAEDecoder.forward 的 numpy 分支）
    static func decodeFrames(decodeIn: Npy, pageAlpha: [Float], meta: TrajMeta,
                             vaeDec: CoreMLModel, transDec: CoreMLModel) throws -> [RGBAImage] {
        let f = decodeIn.shape[0]
        var images: [RGBAImage] = []
        for fr in 0 ..< f {
            let frameSize = 4 * 160 * 160
            let z = Array(decodeIn.data[fr * frameSize ..< (fr + 1) * frameSize])
            let zMA = try MLMultiArray(filledWith: z, shape: [1, 4, 160, 160])
            let pixel = try vaeDec.predict(["z": zMA], output: "var_626").0  // [-1,1]
            var pixel01 = pixel.map { $0 * 0.5 + 0.5 }
            let pxMA = try MLMultiArray(filledWith: pixel01, shape: [1, 3, 1280, 1280])
            let y = try transDec.predict(["pixel": pxMA, "latent": zMA], output: "var_2737").0  // (1,4,1280,1280)
            pixel01 = []  // 释放
            // y: ch0=alpha, ch1..3=premultiplied fg RGB；alpha *= page_alpha；clip[0,1]
            let hw = 1280 * 1280
            var rgba = [UInt8](repeating: 0, count: hw * 4)
            for p in 0 ..< hw {
                let a = min(1, max(0, y[p])) * pageAlpha[p]
                // 对标源码 astype(uint8)：截断而非四舍五入
                rgba[p * 4] = UInt8(min(255, max(0, Int(min(1, max(0, y[hw + p])) * 255))))
                rgba[p * 4 + 1] = UInt8(min(255, max(0, Int(min(1, max(0, y[2 * hw + p])) * 255))))
                rgba[p * 4 + 2] = UInt8(min(255, max(0, Int(min(1, max(0, y[3 * hw + p])) * 255))))
                rgba[p * 4 + 3] = UInt8(min(255, max(0, Int(a * 255))))
            }
            images.append(RGBAImage(w: 1280, h: 1280, px: rgba))
        }
        return images
    }

    /// 逐 tag 与 golden 对比（head 先合成）
    static func compareGolden(images: [RGBAImage], meta: TrajMeta, saveDir: String?) throws -> [[String: Any]] {
        var rows: [[String: Any]] = []
        for (img, tag) in zip(images, meta.tags) {
            let final = meta.composite.map { compositeHeadFrame(img, composite: $0) } ?? img
            if let dir = saveDir { try final.save("\(dir)/\(tag).png") }
            let goldenPath = Paths.golden + "/\(tag).png"
            guard FileManager.default.fileExists(atPath: goldenPath) else { continue }
            let g = try RGBAImage.load(goldenPath)
            let (psnr, maxAbs) = psnrU8(final.px, g.px)
            rows.append(["tag": tag, "psnr_db": psnr, "max_abs": maxAbs, "exact": maxAbs == 0])
            let tagPad = tag + String(repeating: " ", count: max(0, 12 - tag.count))
            print(String(format: "  %@ psnr=%@ dB max_abs=%d %@", tagPad, fmt(psnr), maxAbs, maxAbs == 0 ? "exact" : ""))
        }
        return rows
    }

    static func decode(traj: String, out: String?) throws {
        let meta = try TrajMeta.load(traj)
        try requireFiles([traj + "/decode_in.npy", traj + "/page_alpha.npy"],
                         hint: "golden_trajectory dump 未完成 → 待锚点")
        try requireFiles([Paths.vaeDecoder, Paths.transDecoder], hint: "M2 解码模型缺失")
        let vaeDec = try CoreMLModel(path: Paths.vaeDecoder)
        let transDec = try CoreMLModel(path: Paths.transDecoder)
        let decodeIn = try Npy.load(traj + "/decode_in.npy")
        let pageAlpha = try Npy.load(traj + "/page_alpha.npy").data
        print("== decode \(meta.passName) (F=\(meta.frames)) ==")
        let images = try decodeFrames(decodeIn: decodeIn, pageAlpha: pageAlpha, meta: meta,
                                      vaeDec: vaeDec, transDec: transDec)
        let saveDir = Paths.m3Out + "/decode_\(meta.passName)"
        let rows = try compareGolden(images: images, meta: meta, saveDir: saveDir)
        let pass = rows.allSatisfy { ($0["psnr_db"] as? Double ?? 0) >= 40 }
        print("saved -> \(saveDir)/  (\(rows.allSatisfy { $0["exact"] as? Bool == true } ? "全部逐位一致" : "存在非逐位一致帧"))")
        jsonWrite(["command": "decode", "pass_name": meta.passName, "pass": pass, "tags": rows], out)
        if !pass { exit(1) }
    }

    // MARK: e2e --pass body|head

    static func e2e(pass: String, out: String?) throws {
        let traj = Paths.trajRoot + "/\(pass)"
        let meta = try TrajMeta.load(traj)
        try requireFiles([traj + "/latent_init.npy", traj + "/c_concat.npy", traj + "/vae_moments.npy",
                          traj + "/argb_in.npy", traj + "/page_alpha.npy"],
                         hint: "golden_trajectory dump 未完成 → 待锚点")
        try requireFiles([Paths.vaeEncoder, Paths.vaeDecoder, Paths.transDecoder], hint: "M2 VAE 模型缺失")
        let bundle = try loadUNetBundle(pass: pass, traj: traj)
        let unet = bundle.model
        let statics = bundle.cond
        let vaeEnc = try CoreMLModel(path: Paths.vaeEncoder)
        let vaeDec = try CoreMLModel(path: Paths.vaeDecoder)
        let transDec = try CoreMLModel(path: Paths.transDecoder)

        var result: [String: Any] = ["command": "e2e", "pass_name": pass, "model": bundle.flavor]
        let srcPath = Paths.golden + (pass == "body" ? "/src_img.png" : "/src_head.png")
        let src = try RGBAImage.load(srcPath)
        precondition(src.w == 1280 && src.h == 1280)

        // 1) ARGB 构造（alpha 通道置 1.0 在最前 + RGB，[0,1]）
        let hw = 1280 * 1280
        var argb = [Float](repeating: 0, count: 4 * hw)
        var pageAlpha = [Float](repeating: 0, count: hw)
        for p in 0 ..< hw {
            argb[p] = 1.0
            argb[hw + p] = Float(src.px[p * 4]) / 255
            argb[2 * hw + p] = Float(src.px[p * 4 + 1]) / 255
            argb[3 * hw + p] = Float(src.px[p * 4 + 2]) / 255
            pageAlpha[p] = Float(src.px[p * 4 + 3]) / 255
        }
        let argbAnchor = try Npy.load(traj + "/argb_in.npy").data
        let (psnrArgb, maxAbsArgb) = dynPSNR(ref: argbAnchor, out: argb)
        print(String(format: "argb_in: psnr=%@ dB max_abs=%.3e（期望逐位一致）", fmt(psnrArgb), maxAbsArgb))
        let paAnchor = try Npy.load(traj + "/page_alpha.npy").data
        let (_, maxAbsPA) = dynPSNR(ref: paAnchor, out: pageAlpha)
        print(String(format: "page_alpha: max_abs=%.3e", maxAbsPA))
        result["argb_max_abs"] = maxAbsArgb; result["page_alpha_max_abs"] = maxAbsPA

        // 2) vae_encode: x = (rgb*2-1)*alpha，alpha=1 → rgb*2-1
        var x = [Float](repeating: 0, count: 3 * hw)
        for c in 0 ..< 3 {
            for p in 0 ..< hw { x[c * hw + p] = argb[(c + 1) * hw + p] * 2 - 1 }
        }
        let xMA = try MLMultiArray(filledWith: x, shape: [1, 3, 1280, 1280])
        let t0 = nowMs()
        let moments = try vaeEnc.predict(["x": xMA], output: "var_498").0  // (1,8,160,160)
        print(String(format: "vae encoder forward %.0fms", nowMs() - t0))
        let momentsAnchor = try Npy.load(traj + "/vae_moments.npy").data
        let (psnrMom, maxAbsMom) = dynPSNR(ref: momentsAnchor, out: moments)
        print(String(format: "vae_moments: psnr=%@ dB max_abs=%.3e", fmt(psnrMom), maxAbsMom))
        result["vae_moments_psnr_db"] = psnrMom; result["vae_moments_max_abs"] = maxAbsMom

        // 3) 对齐模式：eps = (c_concat_anchor/scaling - mean)/std，重建验证
        // 注: 本 checkpoint 的 logvar 撞 -30 钳位（std≈4e-5），采样 eps 的影响低于 bf16 量化，
        //     锚点 c_concat ≈ bf16(mean*sf)。因此 eps 反推重建只是管线连通性检查（循环恒等），
        //     真正的编码对齐指标是 mean*sf 与锚点的直接对比（下面一并报告）。
        let ls = 4 * 160 * 160
        let cAnchor = try Npy.load(traj + "/c_concat.npy").data
        let invScale = Float(1.0 / meta.scalingFactor)
        let sf = Float(meta.scalingFactor)
        var cFromMean = [Float](repeating: 0, count: ls)
        var cConcat = [Float](repeating: 0, count: ls)
        for j in 0 ..< ls {
            let mean = moments[j]
            let std = exp(0.5 * moments[ls + j])
            cFromMean[j] = mean * sf
            let eps = (cAnchor[j] * invScale - mean) / std
            cConcat[j] = (mean + std * eps) * sf
        }
        let (psnrM, maxAbsM) = dynPSNR(ref: cAnchor, out: cFromMean)
        print(String(format: "c_concat 直接编码(mean*sf): psnr=%@ dB max_abs=%.3e（真实编码对齐）", fmt(psnrM), maxAbsM))
        let (psnrC, maxAbsC) = dynPSNR(ref: cAnchor, out: cConcat)
        print(String(format: "c_concat 重建: psnr=%@ dB max_abs=%.3e（验证 eps 反推公式/管线连通）", fmt(psnrC), maxAbsC))
        result["c_concat_from_mean_psnr_db"] = psnrM
        result["c_concat_recon_psnr_db"] = psnrC

        // 4) 30 步 CoreML 去噪
        print("== denoise 30 steps ==")
        let (latents, rows, forwardMs) = try denoiseLoop(traj: traj, meta: meta, cConcat: cConcat,
                                                         unet: unet, statics: statics, verbose: true)
        result["denoise_steps"] = rows
        result["forward_ms_total"] = forwardMs.reduce(0, +)
        let decodeRefPath = traj + "/decode_in.npy"
        var decodeInData = latents.map { $0 * invScale }
        if FileManager.default.fileExists(atPath: decodeRefPath) {
            let ref = try Npy.load(decodeRefPath).data
            let (psnr, maxAbs) = dynPSNR(ref: ref, out: decodeInData)
            print(String(format: "decode_in: psnr=%@ dB max_abs=%.4e", fmt(psnr), maxAbs))
            result["decode_in_psnr_db"] = psnr
        }

        // 5) 解码 → PNG → golden 对比（head 走合成）
        let decodeIn = Npy(shape: [meta.frames, 4, 160, 160], data: decodeInData)
        decodeInData = []
        let images = try decodeFrames(decodeIn: decodeIn, pageAlpha: pageAlpha, meta: meta,
                                      vaeDec: vaeDec, transDec: transDec)
        let saveDir = Paths.m3Out + "/e2e_\(pass)"
        print("== golden 对比（\(pass)）==")
        let tagRows = try compareGolden(images: images, meta: meta, saveDir: saveDir)
        result["tags"] = tagRows
        result["golden_min_psnr_db"] = tagRows.compactMap { $0["psnr_db"] as? Double }.min() ?? 0
        print(String(format: "golden min psnr: %@ dB; saved -> %@/", fmt(result["golden_min_psnr_db"] as! Double), saveDir))
        jsonWrite(result, out)
    }
}
