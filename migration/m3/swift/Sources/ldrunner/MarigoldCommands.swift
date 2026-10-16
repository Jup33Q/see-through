import CoreGraphics
import CoreML
import Foundation
import ImageIO

/// Marigold 伪深度 M4 阶段命令。
/// 权威语义：common/utils/inference_utils.py apply_marigold（img_list 装配 /
/// 深度后处理）、common/utils/torchcv.py pad_rgb_torch、
/// common/modules/marigold/marigold_depth_pipeline.py（_np_transform / encode_argb_list /
/// single_infer / decode_depth）、common/utils/cv.py img_alpha_blending(premultiplied=False)。
enum MarigoldCommands {

    /// VALID_BODY_PARTS_V2（19 槽，顺序固定）
    static let slots = ["hair", "headwear", "face", "eyes", "eyewear", "ears", "earwear",
                        "nose", "mouth", "neck", "neckwear", "topwear", "handwear",
                        "bottomwear", "legwear", "footwear", "tail", "wings", "objects"]
    /// compose_list（dict 插入序：eyes 先 hair 后；两槽不相交，顺序无影响）
    static let composeList: [(slot: String, subTags: [String])] = [
        ("eyes", ["eyewhite", "irides", "eyelash", "eyebrow"]),
        ("hair", ["back hair", "front hair"]),
    ]

    // MARK: - 锚点 meta.json

    struct MariMeta {
        var frames: Int
        var tags: [String]
        var depthTags: [String]
        var timesteps: [Int]
        var scalingFactor: Double
        var resolution: Int
        var srcW: Int
        var srcH: Int

        static func load(_ traj: String) throws -> MariMeta {
            let m = try jsonLoad(traj + "/meta.json")
            let hw = (m["src_hw"] as! [Any]).map { ($0 as! NSNumber).intValue }
            return MariMeta(
                frames: (m["frames"] as! NSNumber).intValue,
                tags: m["tags"] as! [String],
                depthTags: m["depth_tags"] as! [String],
                timesteps: (m["timesteps"] as! [Any]).map { Int(($0 as! NSNumber).doubleValue.rounded()) },
                scalingFactor: (m["scaling_factor"] as! NSNumber).doubleValue,
                resolution: (m["resolution"] as! NSNumber).intValue,
                srcW: hw[1], srcH: hw[0])
        }
    }

    // MARK: - L 模式灰度 PNG IO（golden *_depth.png 精确读写）

    static func loadGray(_ path: String) throws -> (w: Int, h: Int, px: [UInt8]) {
        let url = URL(fileURLWithPath: path) as CFURL
        guard let src = CGImageSourceCreateWithURL(url, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { throw Err("cannot read png: \(path)") }
        let w = img.width, h = img.height
        if img.bitsPerPixel == 8, let dp = img.dataProvider, let data = dp.data {
            let bytes = [UInt8](data as Data)
            let bpr = img.bytesPerRow
            var out = [UInt8](repeating: 0, count: w * h)
            for y in 0 ..< h {
                for x in 0 ..< w { out[y * w + x] = bytes[y * bpr + x] }
            }
            return (w, h, out)
        }
        var buf = [UInt8](repeating: 0, count: w * h)
        guard let ctx = CGContext(data: &buf, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: w, space: CGColorSpaceCreateDeviceGray(),
                                  bitmapInfo: 0)
        else { throw Err("cannot create gray ctx for \(path)") }
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
        return (w, h, buf)
    }

    static func saveGray(_ path: String, w: Int, h: Int, px: [UInt8]) throws {
        let data = Data(px)
        guard let provider = CGDataProvider(data: data as CFData),
              let img = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 8,
                                bytesPerRow: w, space: CGColorSpaceCreateDeviceGray(),
                                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                                provider: provider, decode: nil, shouldInterpolate: false,
                                intent: .defaultIntent)
        else { throw Err("gray cgimage create failed") }
        try FileManager.default.createDirectory(
            atPath: URL(fileURLWithPath: path).deletingLastPathComponent().path, withIntermediateDirectories: true)
        guard let dst = CGImageDestinationCreateWithURL(
            URL(fileURLWithPath: path) as CFURL, "public.png" as CFString, 1, nil)
        else { throw Err("dest create failed: \(path)") }
        CGImageDestinationAddImage(dst, img, nil)
        guard CGImageDestinationFinalize(dst) else { throw Err("png write failed: \(path)") }
    }

    // MARK: - 图像装配（apply_marigold img_list 段）

    struct MariAssembly {
        var imgList: [RGBAImage]  // 20 帧 src-res（19 槽 + fullpage）
        var blendedAlpha: [UInt8]
        /// eyes/hair → 存在的子 tag（compose_list 顺序）
        var composeImgs: [String: [(tag: String, img: RGBAImage)]]
    }

    /// img_alpha_blending(premultiplied=False) 非深度/非 xyxy 路径（cv.py:352-470）。
    /// numpy 语义：alpha = uint8/255（float64 提升），累乘在 f64 下进行，末段 clip*255 截断。
    static func alphaBlendStraight(_ imgs: [RGBAImage]) -> RGBAImage {
        let w = imgs[0].w, h = imgs[0].h, n = w * h
        var frgb = [Double](repeating: 0, count: n * 3)
        var fa = [Double](repeating: 0, count: n)
        for img in imgs {
            for p in 0 ..< n {
                let a = Double(img.px[p * 4 + 3]) / 255.0
                fa[p] = min(1.0, max(0.0, fa[p] + a))
                let om = 1.0 - a
                for c in 0 ..< 3 {
                    frgb[p * 3 + c] = frgb[p * 3 + c] * om + Double(img.px[p * 4 + c]) * a
                }
            }
        }
        var out = [UInt8](repeating: 0, count: n * 4)
        for p in 0 ..< n {
            for c in 0 ..< 3 {
                out[p * 4 + c] = UInt8(min(255, max(0, Int(frgb[p * 3 + c]))))
            }
            out[p * 4 + 3] = UInt8(min(255, max(0, Int(min(1.0, max(0.0, fa[p])) * 255.0))))
        }
        return RGBAImage(w: w, h: h, px: out)
    }

    /// apply_marigold img_list 装配：19 槽（alpha<15→0；eyes/hair 由子 tag 合成替换）
    /// + blended_alpha + fullpage（src_img RGB，alpha=blended_alpha）
    static func assembleImgList(golden: String, w: Int, h: Int) throws -> MariAssembly {
        let n = w * h
        let empty = RGBAImage(w: w, h: h, px: [UInt8](repeating: 0, count: n * 4))
        func loadTag(_ tag: String) throws -> RGBAImage? {
            let p = golden + "/\(tag).png"
            guard FileManager.default.fileExists(atPath: p) else { return nil }
            var img = try RGBAImage.load(p)
            precondition(img.w == w && img.h == h, "\(tag).png 尺寸 \(img.w)x\(img.h) != \(w)x\(h)")
            for i in stride(from: 3, to: img.px.count, by: 4) {
                if img.px[i] < 15 { img.px[i] = 0 }
            }
            return img
        }
        var imgList = try slots.map { try loadTag($0) ?? empty }
        var composeImgs: [String: [(tag: String, img: RGBAImage)]] = [:]
        for (c, clist) in composeList {
            var imlist: [(tag: String, img: RGBAImage)] = []
            for tag in clist {
                if let im = try loadTag(tag) { imlist.append((tag: tag, img: im)) }
            }
            if !imlist.isEmpty {
                imgList[slots.firstIndex(of: c)!] = alphaBlendStraight(imlist.map(\.img))
                composeImgs[c] = imlist
            }
        }
        // blended_alpha = 19 帧 alpha 逐像素 fp32 累加 clip(0,1)*255→uint8（截断）
        var acc = [Float](repeating: 0, count: n)
        for img in imgList {
            for p in 0 ..< n { acc[p] += Float(img.px[p * 4 + 3]) / 255 }
        }
        var blended = [UInt8](repeating: 0, count: n)
        for p in 0 ..< n {
            blended[p] = UInt8(min(255, max(0, Int(min(1, max(0, acc[p])) * 255))))
        }
        var full = try RGBAImage.load(golden + "/src_img.png")
        precondition(full.w == w && full.h == h)
        for p in 0 ..< n { full.px[p * 4 + 3] = blended[p] }
        imgList.append(full)
        return MariAssembly(imgList: imgList, blendedAlpha: blended, composeImgs: composeImgs)
    }

    // MARK: - float resize / pad_rgb_torch

    /// torch F.interpolate bilinear align_corners=False / cv2 INTER_LINEAR float 语义
    /// （中心对齐映射，Double 位置 Float 权重），CHW 多通道平面
    static func resizeFloatBilinear(_ src: [Float], w: Int, h: Int, tw: Int, th: Int, channels: Int) -> [Float] {
        if tw == w && th == h { return src }
        let sx = Double(w) / Double(tw), sy = Double(h) / Double(th)
        var out = [Float](repeating: 0, count: channels * th * tw)
        for dy in 0 ..< th {
            let fy = max(0.0, (Double(dy) + 0.5) * sy - 0.5)
            let y0 = min(Int(fy), h - 1), y1 = min(y0 + 1, h - 1)
            let wy = Float(fy - Double(y0))
            for dx in 0 ..< tw {
                let fx = max(0.0, (Double(dx) + 0.5) * sx - 0.5)
                let x0 = min(Int(fx), w - 1), x1 = min(x0 + 1, w - 1)
                let wx = Float(fx - Double(x0))
                for c in 0 ..< channels {
                    let base = c * h * w
                    let v00 = src[base + y0 * w + x0], v01 = src[base + y0 * w + x1]
                    let v10 = src[base + y1 * w + x0], v11 = src[base + y1 * w + x1]
                    let top = v00 * (1 - wx) + v01 * wx
                    let bot = v10 * (1 - wx) + v11 * wx
                    out[(c * th + dy) * tw + dx] = top * (1 - wy) + bot * wy
                }
            }
        }
        return out
    }

    /// torchcv.py:24 pad_rgb_torch（input_format='argb', return_format='argb' 后取 rgb）：
    /// build_alpha_pyramid_torch 预乘 alpha、dk=1.2 金字塔（bilinear align_corners=False），
    /// 自最小层向上 fg = layer_c + fg*(1-layer_a)。输入 argb (4,h,w) [0,1]，输出 rgb (3,h,w)。
    static func padRgbTorch(_ argb: [Float], w: Int, h: Int) -> [Float] {
        let n = w * h
        var cur = argb
        for c in 1 ... 3 {  // 预乘 alpha（ch0 不动，逐 channel 安全）
            for p in 0 ..< n { cur[c * n + p] *= cur[p] }
        }
        var pyramid: [(data: [Float], w: Int, h: Int)] = []
        var (cw, ch) = (w, h)
        while true {
            pyramid.append((cur, cw, ch))
            if min(cw, ch) == 1 { break }
            let nw = Int(Double(cw) / 1.2), nh = Int(Double(ch) / 1.2)  // int(W/dk) 截断
            cur = resizeFloatBilinear(cur, w: cw, h: ch, tw: nw, th: nh, channels: 4)
            (cw, ch) = (nw, nh)
        }
        // fg = sum(top_c) / clip(sum(top_a), 1e-8, 1e32)（1×1 即元素自身）
        let top = pyramid.last!
        let denom = min(max(top.data[0], Float(1e-8)), Float(1e32))
        var fg = [Float](repeating: 0, count: 3)
        for c in 0 ..< 3 { fg[c] = top.data[c + 1] / denom }
        var (fw, fh) = (top.w, top.h)
        for layer in pyramid.reversed() {
            fg = resizeFloatBilinear(fg, w: fw, h: fh, tw: layer.w, th: layer.h, channels: 3)
            (fw, fh) = (layer.w, layer.h)
            let ln = fw * fh
            for c in 0 ..< 3 {
                for p in 0 ..< ln {
                    fg[c * ln + p] = layer.data[(c + 1) * ln + p] + fg[c * ln + p] * (Float(1.0) - layer.data[p])
                }
            }
        }
        return fg
    }

    // MARK: - 深度后处理（inference_utils.py:286-320，normalize_depth=False）

    /// depthSrc: 20 帧 src-res float depth（仅前 19 帧落盘）。
    /// 返回 [(depth_tag, px)]，顺序与 meta.depthTags 一致。
    static func depthPostprocess(depthSrc: [[Float]], asm: MariAssembly, w: Int, h: Int) -> [(tag: String, px: [UInt8])] {
        let n = w * h
        var out: [(String, [UInt8])] = []
        for ii in 0 ..< slots.count {
            // depth_u8 = (clip(d,0,1)*255).astype(uint8)（截断非四舍五入）
            var du8 = [UInt8](repeating: 0, count: n)
            for p in 0 ..< n {
                du8[p] = UInt8(min(255, max(0, Int(Double(min(1, max(0, depthSrc[ii][p]))) * 255.0))))
            }
            let tag = slots[ii]
            guard let comp = asm.composeImgs[tag] else {
                out.append((tag, du8))
                continue
            }
            // mask 初值 blended_alpha > 256：uint8 恒 False（忠实移植）
            var mask = [Bool](repeating: false, count: n)
            for (t, im) in comp.reversed() {
                var maskLocal = [Bool](repeating: false, count: n)
                for p in 0 ..< n { maskLocal[p] = im.px[p * 4 + 3] > 15 }
                var dl = [UInt8](repeating: 255, count: n)
                var invis: [Int] = []
                var vis: [Int] = []
                for p in 0 ..< n where maskLocal[p] {
                    dl[p] = du8[p]
                    if mask[p] { invis.append(p) } else { vis.append(p) }
                }
                if !invis.isEmpty {
                    // depth_local[mask_invis] = median(depth[mask_local & ~mask_invis])
                    // np.median 偶数取中两值均值（f64），赋入 uint8 数组截断
                    let vals = vis.map { du8[$0] }.sorted()
                    let med: Double = vals.isEmpty ? 0 : (vals.count % 2 == 1
                        ? Double(vals[vals.count / 2])
                        : (Double(vals[vals.count / 2 - 1]) + Double(vals[vals.count / 2])) / 2)
                    let medU8 = UInt8(min(255, max(0, Int(med))))
                    for p in invis { dl[p] = medU8 }
                }
                for p in 0 ..< n { mask[p] = mask[p] || maskLocal[p] }
                out.append((t, dl))
            }
        }
        return out
    }

    // MARK: - UNet / 去噪循环

    static func loadEhs(_ traj: String, frames: Int) throws -> [Float] {
        let ehs = try Npy.load(traj + "/empty_text_embed.npy")  // (1,2,1024)
        return expandFrames(ehs.data, frames: frames)
    }

    static func mariUnetInputs(sample: [Float], timestep: Int, ehs: [Float], frames: Int) throws -> [String: MLMultiArray] {
        [
            "sample": try MLMultiArray(filledWith: sample, shape: [1, frames, 12, 96, 96]),
            "timestep": try MLMultiArray(filledWith: [Float(timestep)], shape: [1]),
            "encoder_hidden_states": try MLMultiArray(filledWith: ehs, shape: [frames, 2, 1024]),
        ]
    }

    /// 4 步 DDIM 去噪循环（latent_init + cond_latent 锚点 + CoreML UNet，调度器实时算）。
    /// unet_in 拼装 = cat([cond_latent[f], latents[f]], ch) → [1,F,12,96,96]。
    static func mariDenoiseLoop(traj: String, meta: MariMeta, condLatent: [Float],
                                unet: CoreMLModel, ehs: [Float], verbose: Bool) throws
        -> (latents: [Float], rows: [[String: Any]], forwardMs: [Double]) {
        let f = meta.frames
        let plane = 96 * 96
        let latentSize = 4 * plane
        let condSize = 8 * plane
        var latents = try Npy.load(traj + "/latent_init.npy").data
        let sched = DDIMScheduler()
        sched.setTimesteps(4)
        precondition(sched.timesteps == meta.timesteps, "DDIM timesteps 与锚点 meta 不一致")
        var rows: [[String: Any]] = []
        var forwardMs: [Double] = []
        for i in 0 ..< 4 {
            var input = [Float](repeating: 0, count: f * 12 * plane)
            for fr in 0 ..< f {
                input.replaceSubrange((fr * 12) * plane ..< (fr * 12 + 8) * plane,
                                      with: condLatent[fr * condSize ..< (fr + 1) * condSize])
                input.replaceSubrange((fr * 12 + 8) * plane ..< (fr * 12 + 12) * plane,
                                      with: latents[fr * latentSize ..< (fr + 1) * latentSize])
            }
            let inputs = try mariUnetInputs(sample: input, timestep: meta.timesteps[i], ehs: ehs, frames: f)
            let t0 = nowMs()
            let pred = try unet.predict(inputs, output: "noise_pred").0
            let ms = nowMs() - t0
            forwardMs.append(ms)
            latents = sched.step(modelOutput: pred, timestep: meta.timesteps[i], sample: latents)
            var row: [String: Any] = ["step": i, "timestep": meta.timesteps[i], "forward_ms": ms]
            let refPath = traj + String(format: "/step_%02d_latents_next.npy", i)
            if FileManager.default.fileExists(atPath: refPath) {
                let ref = try Npy.load(refPath).data
                let (psnr, maxAbs) = dynPSNR(ref: ref, out: latents)
                row["psnr_db"] = psnr; row["max_abs"] = maxAbs
                if verbose {
                    print(String(format: "  step %02d (t=%d): forward=%.0fms latents psnr=%@ dB max_abs=%.4e",
                                 i, meta.timesteps[i], ms, fmt(psnr), maxAbs))
                }
            } else if verbose {
                print(String(format: "  step %02d (t=%d): forward=%.0fms", i, meta.timesteps[i], ms))
            }
            rows.append(row)
        }
        return (latents, rows, forwardMs)
    }

    // MARK: - 解码 + 后处理共用段

    /// decode_in 逐帧过 marigold_vae_decoder → 原始 decode_out（[-1,1] 域）
    static func decodeDepth(decodeIn: Npy, vaeDec: CoreMLModel) throws -> (decodeOut: [Float], depth768: [Float]) {
        let f = decodeIn.shape[0]
        let frameSize = 4 * 96 * 96
        var decodeOut = [Float]()
        var depth768 = [Float]()
        decodeOut.reserveCapacity(f * 768 * 768)
        for fr in 0 ..< f {
            let z = Array(decodeIn.data[fr * frameSize ..< (fr + 1) * frameSize])
            let zMA = try MLMultiArray(filledWith: z, shape: [1, 4, 96, 96])
            let d = try vaeDec.predict(["z": zMA]).0  // (1,1,768,768)，模型内部已 /0.18215
            decodeOut.append(contentsOf: d)
            // clip(-1,1) 后 (d+1)/2（fp32，对标 dump 脚本 depth768 重算）
            for v in d { depth768.append((min(1, max(-1, v)) + 1) / 2) }
        }
        return (decodeOut, depth768)
    }

    /// depth768（20×768² [0,1]）→ resize 回 src → 23 tag 深度 PNG → vs golden。
    /// 返回 (tag 对比行, 锚点对比附注)
    static func postprocessCompare(depth768: [Float], meta: MariMeta, traj: String,
                                   saveDir: String) throws -> (rows: [[String: Any]], extras: [String: Any]) {
        let f = meta.frames
        let n768 = 768 * 768
        var extras: [String: Any] = [:]
        // resize 768→src（float 双线性，cv2 INTER_LINEAR 语义）
        var depthSrc = [Float]()
        depthSrc.reserveCapacity(f * meta.srcW * meta.srcH)
        for fr in 0 ..< f {
            let frame = Array(depth768[fr * n768 ..< (fr + 1) * n768])
            depthSrc.append(contentsOf: resizeFloatBilinear(frame, w: 768, h: 768,
                                                            tw: meta.srcW, th: meta.srcH, channels: 1))
        }
        let srcAnchorPath = traj + "/depth_pred_src.npy"
        if FileManager.default.fileExists(atPath: srcAnchorPath) {
            let ref = try Npy.load(srcAnchorPath).data
            let (psnr, maxAbs) = dynPSNR(ref: ref, out: depthSrc)
            print(String(format: "depth_pred_src: psnr=%@ dB max_abs=%.4e", fmt(psnr), maxAbs))
            extras["depth_pred_src_psnr_db"] = psnr; extras["depth_pred_src_max_abs"] = maxAbs
        }
        // 后处理（mask 语义需要 src-res 装配）
        let asm = try assembleImgList(golden: Paths.golden, w: meta.srcW, h: meta.srcH)
        var frames: [[Float]] = []
        for fr in 0 ..< f {
            frames.append(Array(depthSrc[fr * meta.srcW * meta.srcH ..< (fr + 1) * meta.srcW * meta.srcH]))
        }
        let pngs = depthPostprocess(depthSrc: frames, asm: asm, w: meta.srcW, h: meta.srcH)
        var rows: [[String: Any]] = []
        for (tag, px) in pngs {
            try saveGray("\(saveDir)/\(tag)_depth.png", w: meta.srcW, h: meta.srcH, px: px)
            let gp = Paths.golden + "/\(tag)_depth.png"
            guard FileManager.default.fileExists(atPath: gp) else { continue }
            let g = try loadGray(gp)
            let (psnr, maxAbs) = psnrU8(px, g.px)
            rows.append(["tag": tag, "psnr_db": psnr, "max_abs": maxAbs, "exact": maxAbs == 0])
            let tagPad = tag + String(repeating: " ", count: max(0, 12 - tag.count))
            print(String(format: "  %@ psnr=%@ dB max_abs=%d %@", tagPad, fmt(psnr), maxAbs, maxAbs == 0 ? "exact" : ""))
        }
        precondition(pngs.map(\.tag) == meta.depthTags, "depth tag 顺序与 meta.depth_tags 不一致")
        return (rows, extras)
    }

    // MARK: mari-sched-check-fixture（无需锚点/模型）

    static func schedCheckFixture(out: String?) throws {
        let sched = DDIMScheduler()
        sched.setTimesteps(4)
        let ref = try jsonLoad(Paths.fixtures + "/ddim_schedule_ref.json")
        let refTs = (ref["timesteps"] as! [Any]).map { ($0 as! NSNumber).intValue }
        let refAC = (ref["alphas_cumprod"] as! [Any]).map { ($0 as! NSNumber).doubleValue }
        let refFinal = (ref["final_alpha_cumprod"] as! NSNumber).doubleValue

        let tsOK = sched.timesteps == refTs
        var maxRelAC = 0.0
        var bitExact = 0
        for (a, b) in zip(sched.alphasCumprod, refAC) {
            let denom = max(abs(b), 1e-30)
            maxRelAC = max(maxRelAC, abs(Double(a) - b) / denom)
            if Double(a) == b { bitExact += 1 }
        }
        let acOK = maxRelAC <= 1e-6
        let finalOK = Double(sched.finalAlphaCumprod) == refFinal
        print("== ddim_schedule_ref ==")
        print("timesteps match: \(tsOK)  \(sched.timesteps)")
        print(String(format: "alphas_cumprod max_rel_err: %.3e (tol 1e-6), 逐位一致 %d/1000", maxRelAC, bitExact))
        print("final_alpha_cumprod: \(sched.finalAlphaCumprod) (ref \(refFinal)) exact: \(finalOK)")

        // ddim_steps_ref：确定性逐步回放（eta=0，v_prediction，fp32 对 fp32）
        let sr = try jsonLoad(Paths.fixtures + "/ddim_steps_ref.json")
        var sample = (sr["sample0"] as! [Any]).map { Float(($0 as! NSNumber).doubleValue) }
        let mos = sr["model_outputs"] as! [Any]
        let refs = sr["latents_next"] as! [Any]
        let ts = (sr["timesteps"] as! [Any]).map { ($0 as! NSNumber).intValue }
        precondition(mos.count == 4 && ts == refTs)

        let sched2 = DDIMScheduler()
        sched2.setTimesteps(4)
        var rows: [[String: Any]] = []
        var maxAbsAll = 0.0, minPsnr = Double.infinity
        print("== ddim_steps_ref replay (4 steps) ==")
        for i in 0 ..< 4 {
            let mo = (mos[i] as! [Any]).map { Float(($0 as! NSNumber).doubleValue) }
            let next = sched2.step(modelOutput: mo, timestep: ts[i], sample: sample)
            let refNext = (refs[i] as! [Any]).map { Float(($0 as! NSNumber).doubleValue) }
            let (psnr, maxAbs) = dynPSNR(ref: refNext, out: next)
            maxAbsAll = max(maxAbsAll, maxAbs); minPsnr = min(minPsnr, psnr)
            rows.append(["step": i, "timestep": ts[i], "psnr_db": psnr, "max_abs": maxAbs])
            print(String(format: "  step %d (t=%d): psnr=%@ dB max_abs=%.3e", i, ts[i], fmt(psnr), maxAbs))
            sample = next
        }
        let replayOK = maxAbsAll < 1e-5
        print(String(format: "replay summary: max_abs=%.3e min_psnr=%@ dB → %@", maxAbsAll, fmt(minPsnr), replayOK ? "PASS" : "FAIL"))
        let pass = tsOK && acOK && finalOK && replayOK
        jsonWriteM4([
            "command": "mari-sched-check-fixture", "pass": pass,
            "timesteps_match": tsOK, "alphas_cumprod_max_rel_err": maxRelAC,
            "alphas_cumprod_bitexact": bitExact, "final_alpha_cumprod_exact": finalOK,
            "replay_max_abs": maxAbsAll, "replay_min_psnr_db": minPsnr, "steps": rows,
        ], out)
        print(pass ? "PASS" : "FAIL")
        if !pass { exit(1) }
    }

    // MARK: mari-sched-check（锚点逐步对齐，eta=0 确定性直接对比）

    static func schedCheck(traj: String, out: String?) throws {
        let meta = try MariMeta.load(traj)
        try requireFiles([traj + "/latent_init.npy", traj + "/step_00_unet_out.npy", traj + "/step_00_latents_next.npy"],
                         hint: "golden_trajectory dump 未完成 → 待锚点")
        let sched = DDIMScheduler()
        sched.setTimesteps(4)
        precondition(sched.timesteps == meta.timesteps, "DDIM timesteps 与锚点 meta 不一致")
        print("== mari-sched-check (F=\(meta.frames), timesteps=\(meta.timesteps)) ==")
        var rows: [[String: Any]] = []
        var maxAbsAll = 0.0, minPsnr = Double.infinity
        for i in 0 ..< 4 {
            let sample: [Float] = try i == 0
                ? Npy.load(traj + "/latent_init.npy").data
                : Npy.load(traj + String(format: "/step_%02d_latents_next.npy", i - 1)).data
            let mo = try Npy.load(traj + String(format: "/step_%02d_unet_out.npy", i)).data
            let ref = try Npy.load(traj + String(format: "/step_%02d_latents_next.npy", i)).data
            let next = sched.step(modelOutput: mo, timestep: meta.timesteps[i], sample: sample)
            let (psnr, maxAbs) = dynPSNR(ref: ref, out: next)
            maxAbsAll = max(maxAbsAll, maxAbs); minPsnr = min(minPsnr, psnr)
            rows.append(["step": i, "timestep": meta.timesteps[i], "psnr_db": psnr, "max_abs": maxAbs])
            print(String(format: "  step %d (t=%d): psnr=%@ dB max_abs=%.4e", i, meta.timesteps[i], fmt(psnr), maxAbs))
        }
        // PASS 阈值依据：锚点是 torch MPS bf16 逐步算术（存 fp32），bf16 相对量化 ~2^-8；
        // latents 幅度 ~±5，单步若干次 bf16 舍入累积上限 ~0.03。阈值 0.05 同 M3 sched-check
        // 末步的 bf16 量化论证；fp32 复刻实测应 ≤1e-3 量级。
        let pass = maxAbsAll < 0.05
        print(String(format: "summary: max_abs=%.3e min_psnr=%@ dB (阈值 max_abs<0.05, bf16 量化论证) → %@",
                     maxAbsAll, fmt(minPsnr), pass ? "PASS" : "FAIL"))
        jsonWriteM4(["command": "mari-sched-check", "pass": pass,
                     "max_abs": maxAbsAll, "min_psnr_db": minPsnr, "steps": rows], out)
        if !pass { exit(1) }
    }

    // MARK: mari-unet-step --step N

    static func unetStep(traj: String, stepN: Int, out: String?) throws {
        let meta = try MariMeta.load(traj)
        try requireFiles([traj + String(format: "/step_%02d_unet_in.npy", stepN),
                          traj + String(format: "/step_%02d_unet_out.npy", stepN),
                          traj + "/empty_text_embed.npy"], hint: "golden_trajectory dump 未完成 → 待锚点")
        try requireFiles([Paths.mariUnet], hint: "M4 UNet 转换未完成 → 待模型")
        let unet = try CoreMLModel(path: Paths.mariUnet)
        let sampleNpy = try Npy.load(traj + String(format: "/step_%02d_unet_in.npy", stepN))
        let refNpy = try Npy.load(traj + String(format: "/step_%02d_unet_out.npy", stepN))
        let ehs = try loadEhs(traj, frames: meta.frames)

        let inputs = try mariUnetInputs(sample: sampleNpy.data, timestep: meta.timesteps[stepN],
                                        ehs: ehs, frames: meta.frames)
        var pred = [Float]()
        var times: [Double] = []
        for r in 0 ..< 3 {
            let t0 = nowMs()
            pred = try unet.predict(inputs, output: "noise_pred").0
            times.append(nowMs() - t0)
            if r == 0 { print("[mari-unet-step] first forward \(fmt(times[0], 0))ms（含编译/缓存）") }
        }
        let (psnr, maxAbs) = dynPSNR(ref: refNpy.data, out: pred)
        let pass = psnr >= 60  // 参考 M4 转换对齐：step0≈61.7dB, step2≈63.9dB
        print(String(format: "mari-unet-step %02d (t=%d): psnr=%@ dB max_abs=%.4e (阈值 60dB → %@) | forward ms: first=%.0f warm=%@",
                     stepN, meta.timesteps[stepN], fmt(psnr), maxAbs, pass ? "PASS" : "FAIL",
                     times[0], times.dropFirst().map { fmt($0, 0) }.joined(separator: ",")))
        jsonWriteM4(["command": "mari-unet-step", "step": stepN, "timestep": meta.timesteps[stepN],
                     "pass": pass, "psnr_db": psnr, "max_abs": maxAbs,
                     "forward_ms_first": times[0], "forward_ms_warm": Array(times.dropFirst())], out)
        if !pass { exit(1) }
    }

    // MARK: mari-denoise（latent_init + cond_latent 锚点 + CoreML UNet 4 步循环）

    static func denoise(traj: String, out: String?) throws {
        let meta = try MariMeta.load(traj)
        try requireFiles([traj + "/latent_init.npy", traj + "/cond_latent.npy", traj + "/empty_text_embed.npy"],
                         hint: "golden_trajectory dump 未完成 → 待锚点")
        try requireFiles([Paths.mariUnet], hint: "M4 UNet 转换未完成 → 待模型")
        let unet = try CoreMLModel(path: Paths.mariUnet)
        let condLatent = try Npy.load(traj + "/cond_latent.npy").data
        let ehs = try loadEhs(traj, frames: meta.frames)
        print("== mari-denoise (F=\(meta.frames), 4 steps, CoreML marigold_unet_fp16) ==")
        let (latents, rows, forwardMs) = try mariDenoiseLoop(traj: traj, meta: meta, condLatent: condLatent,
                                                             unet: unet, ehs: ehs, verbose: true)
        var result: [String: Any] = ["command": "mari-denoise", "steps": rows,
                                     "forward_ms_total": forwardMs.reduce(0, +)]
        // 末步 latents 直接对 decode_in（锚点 decode_in == step_03_latents_next，未缩放）
        let ref = try Npy.load(traj + "/decode_in.npy").data
        let (psnr, maxAbs) = dynPSNR(ref: ref, out: latents)
        print(String(format: "decode_in: psnr=%@ dB max_abs=%.4e", fmt(psnr), maxAbs))
        result["decode_in_psnr_db"] = psnr; result["decode_in_max_abs"] = maxAbs
        print(String(format: "CoreML forward total %.1fs (avg %.0fms/step)",
                     forwardMs.reduce(0, +) / 1000, forwardMs.reduce(0, +) / 4))
        jsonWriteM4(result, out)
    }

    // MARK: mari-decode（锚点 decode_in → 23 深度 PNG → golden）

    static func decode(traj: String, out: String?) throws {
        let meta = try MariMeta.load(traj)
        try requireFiles([traj + "/decode_in.npy", traj + "/decode_out.npy", traj + "/depth_pred_768.npy"],
                         hint: "golden_trajectory dump 未完成 → 待锚点")
        try requireFiles([Paths.mariVaeDecoder], hint: "M4 VAE decoder 转换未完成 → 待模型")
        let vaeDec = try CoreMLModel(path: Paths.mariVaeDecoder)
        let decodeIn = try Npy.load(traj + "/decode_in.npy")
        print("== mari-decode (F=\(meta.frames)) ==")
        var result: [String: Any] = ["command": "mari-decode"]
        let t0 = nowMs()
        let (decodeOut, depth768) = try decodeDepth(decodeIn: decodeIn, vaeDec: vaeDec)
        print(String(format: "vae decoder ×%d total %.1fs", meta.frames, (nowMs() - t0) / 1000))
        let outAnchor = try Npy.load(traj + "/decode_out.npy").data
        let (psnrOut, maxAbsOut) = dynPSNR(ref: outAnchor, out: decodeOut)
        print(String(format: "decode_out: psnr=%@ dB max_abs=%.4e", fmt(psnrOut), maxAbsOut))
        result["decode_out_psnr_db"] = psnrOut; result["decode_out_max_abs"] = maxAbsOut
        let d768Anchor = try Npy.load(traj + "/depth_pred_768.npy").data
        let (psnr768, maxAbs768) = dynPSNR(ref: d768Anchor, out: depth768)
        print(String(format: "depth_pred_768: psnr=%@ dB max_abs=%.4e", fmt(psnr768), maxAbs768))
        result["depth_pred_768_psnr_db"] = psnr768; result["depth_pred_768_max_abs"] = maxAbs768

        let saveDir = Paths.m4Out + "/decode_marigold"
        let (rows, extras) = try postprocessCompare(depth768: depth768, meta: meta, traj: traj, saveDir: saveDir)
        result["tags"] = rows
        for (k, v) in extras { result[k] = v }
        let minPsnr = rows.compactMap { $0["psnr_db"] as? Double }.min() ?? 0
        let nExact = rows.filter { $0["exact"] as? Bool == true }.count
        result["golden_min_psnr_db"] = minPsnr; result["golden_exact"] = nExact
        let pass = minPsnr >= 35  // 验收阈值（bf16 锚点 + fp16 CoreML，预期 50+）
        result["pass"] = pass
        print(String(format: "golden: min psnr=%@ dB, exact %d/%d; saved -> %@/ → %@",
                     fmt(minPsnr), nExact, rows.count, saveDir, pass ? "PASS" : "FAIL"))
        jsonWriteM4(result, out)
        if !pass { exit(1) }
    }

    // MARK: mari-e2e（golden tag PNG → 全胶水 → 23 深度 PNG vs golden）

    static func e2e(traj: String, out: String?) throws {
        let meta = try MariMeta.load(traj)
        try requireFiles([traj + "/argb_pre_pad.npy", traj + "/vae_rgb_in.npy", traj + "/vae_latents.npy",
                          traj + "/cond_latent.npy", traj + "/latent_init.npy", traj + "/empty_text_embed.npy",
                          traj + "/decode_in.npy"],
                         hint: "golden_trajectory dump 未完成 → 待锚点")
        try requireFiles([Paths.mariUnet, Paths.mariVaeEncoder, Paths.mariVaeDecoder],
                         hint: "M4 模型转换未完成 → 待模型")
        let res = meta.resolution
        let nRes = res * res
        var result: [String: Any] = ["command": "mari-e2e"]

        // 1) img_list 装配（src-res）→ smartResize 768 → ARGB/255
        print("== 装配 + resize \(meta.srcW)²→\(res)² ==")
        let asm = try assembleImgList(golden: Paths.golden, w: meta.srcW, h: meta.srcH)
        var argb = [Float]()
        argb.reserveCapacity(21 * 4 * nRes)
        for img in asm.imgList {
            let r = smartResize(img, res, res)  // 1280²→768² 降采样 INTER_AREA
            for c in [3, 0, 1, 2] {  // ARGB（alpha 在最前）
                for p in 0 ..< nRes { argb.append(Float(r.px[p * 4 + c]) / 255) }
            }
        }
        // entry 20 = cond_full_page（=entry 19，与锚点一致）
        argb.append(contentsOf: argb[19 * 4 * nRes ..< 20 * 4 * nRes])
        let argbAnchor = try Npy.load(traj + "/argb_pre_pad.npy").data
        let (psnrArgb, maxAbsArgb) = dynPSNR(ref: argbAnchor, out: argb)
        print(String(format: "argb_pre_pad: psnr=%@ dB max_abs=%.3e（期望接近逐位）", fmt(psnrArgb), maxAbsArgb))
        result["argb_pre_pad_psnr_db"] = psnrArgb; result["argb_pre_pad_max_abs"] = maxAbsArgb

        // 2) pad_rgb_torch → rgb*2-1
        var rgbIn = [Float]()
        rgbIn.reserveCapacity(21 * 3 * nRes)
        for e in 0 ..< 21 {
            let frame = Array(argb[e * 4 * nRes ..< (e + 1) * 4 * nRes])
            let padded = padRgbTorch(frame, w: res, h: res)
            for v in padded { rgbIn.append(v * 2 - 1) }
        }
        let rgbAnchor = try Npy.load(traj + "/vae_rgb_in.npy").data
        let (psnrRgb, maxAbsRgb) = dynPSNR(ref: rgbAnchor, out: rgbIn)
        print(String(format: "vae_rgb_in: psnr=%@ dB max_abs=%.3e", fmt(psnrRgb), maxAbsRgb))
        result["vae_rgb_in_psnr_db"] = psnrRgb; result["vae_rgb_in_max_abs"] = maxAbsRgb
        // 3) CoreML VAE enc ×21（entry 20 = cond_full_page 独立前向，忠实锚点结构）
        let vaeEnc = try CoreMLModel(path: Paths.mariVaeEncoder)
        var vaeLatents = [Float]()
        vaeLatents.reserveCapacity(21 * 4 * 96 * 96)
        let t0 = nowMs()
        for e in 0 ..< 21 {
            let x = Array(rgbIn[e * 3 * nRes ..< (e + 1) * 3 * nRes])
            let xMA = try MLMultiArray(filledWith: x, shape: [1, 3, res, res])
            let lat = try vaeEnc.predict(["rgb": xMA]).0  // (1,4,96,96) scaled latent
            vaeLatents.append(contentsOf: lat)
        }
        print(String(format: "vae encoder ×21 total %.1fs", (nowMs() - t0) / 1000))
        let latAnchor = try Npy.load(traj + "/vae_latents.npy").data
        let (psnrLat, maxAbsLat) = dynPSNR(ref: latAnchor, out: vaeLatents)
        print(String(format: "vae_latents: psnr=%@ dB max_abs=%.3e", fmt(psnrLat), maxAbsLat))
        result["vae_latents_psnr_db"] = psnrLat; result["vae_latents_max_abs"] = maxAbsLat
        argb = []
        rgbIn = []

        // 4) cond_latent 拼装：cat([page_latent(entry20).expand, frame_latents], ch)
        let plane = 96 * 96
        let ls = 4 * plane
        var condLatent = [Float](repeating: 0, count: 20 * 8 * plane)
        let page = vaeLatents[20 * ls ..< 21 * ls]
        for fr in 0 ..< 20 {
            condLatent.replaceSubrange((fr * 8) * plane ..< (fr * 8 + 4) * plane, with: page)
            condLatent.replaceSubrange((fr * 8 + 4) * plane ..< (fr * 8 + 8) * plane,
                                       with: vaeLatents[fr * ls ..< (fr + 1) * ls])
        }
        let condAnchor = try Npy.load(traj + "/cond_latent.npy").data
        let (psnrCond, maxAbsCond) = dynPSNR(ref: condAnchor, out: condLatent)
        print(String(format: "cond_latent: psnr=%@ dB max_abs=%.3e", fmt(psnrCond), maxAbsCond))
        result["cond_latent_psnr_db"] = psnrCond; result["cond_latent_max_abs"] = maxAbsCond

        // 5) latent_init 取锚点（对齐模式：DDIM 无 RNG，唯一外部随机源即初始噪声；
        //    mariDenoiseLoop 内部自锚点加载）

        // 6) 4 步 CoreML 去噪
        let unet = try CoreMLModel(path: Paths.mariUnet)
        let ehs = try loadEhs(traj, frames: meta.frames)
        print("== denoise 4 steps ==")
        let (latents, rows, forwardMs) = try mariDenoiseLoop(traj: traj, meta: meta, condLatent: condLatent,
                                                             unet: unet, ehs: ehs, verbose: true)
        result["denoise_steps"] = rows
        result["forward_ms_total"] = forwardMs.reduce(0, +)
        let decRef = try Npy.load(traj + "/decode_in.npy").data
        let (psnrDI, maxAbsDI) = dynPSNR(ref: decRef, out: latents)
        print(String(format: "decode_in: psnr=%@ dB max_abs=%.4e", fmt(psnrDI), maxAbsDI))
        result["decode_in_psnr_db"] = psnrDI; result["decode_in_max_abs"] = maxAbsDI

        // 7) 解码 → 768 → src → 后处理 → golden
        let vaeDec = try CoreMLModel(path: Paths.mariVaeDecoder)
        let decodeIn = Npy(shape: [meta.frames, 4, 96, 96], data: latents)
        let t1 = nowMs()
        let (_, depth768) = try decodeDepth(decodeIn: decodeIn, vaeDec: vaeDec)
        print(String(format: "vae decoder ×%d total %.1fs", meta.frames, (nowMs() - t1) / 1000))
        let d768Anchor = try Npy.load(traj + "/depth_pred_768.npy").data
        let (psnr768, maxAbs768) = dynPSNR(ref: d768Anchor, out: depth768)
        print(String(format: "depth_pred_768: psnr=%@ dB max_abs=%.4e", fmt(psnr768), maxAbs768))
        result["depth_pred_768_psnr_db"] = psnr768; result["depth_pred_768_max_abs"] = maxAbs768

        let saveDir = Paths.m4Out + "/e2e_marigold"
        print("== golden 对比（23 depth PNG）==")
        let (tagRows, extras) = try postprocessCompare(depth768: depth768, meta: meta, traj: traj, saveDir: saveDir)
        result["tags"] = tagRows
        for (k, v) in extras { result[k] = v }
        let minPsnr = tagRows.compactMap { $0["psnr_db"] as? Double }.min() ?? 0
        let nExact = tagRows.filter { $0["exact"] as? Bool == true }.count
        result["golden_min_psnr_db"] = minPsnr; result["golden_exact"] = nExact
        let pass = minPsnr >= 35  // 验收阈值 35dB（预期 50+）
        result["pass"] = pass
        print(String(format: "golden: min psnr=%@ dB, exact %d/%d; saved -> %@/ → %@",
                     fmt(minPsnr), nExact, tagRows.count, saveDir, pass ? "PASS" : "FAIL"))
        jsonWriteM4(result, out)
        if !pass { exit(1) }
    }
}
