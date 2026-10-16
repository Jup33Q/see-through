import Foundation

/// DDIMScheduler 的 Swift 精确移植。
/// 权威来源: diffusers 0.37.0 scheduling_ddim.py（rescale_zero_terminal_snr 同
/// scheduling_euler_discrete.py:107）。
/// 固定 config（marigold scheduler_config.json）:
///   num_train_timesteps=1000, beta_schedule=scaled_linear, beta_start=0.00085,
///   beta_end=0.012, prediction_type=v_prediction, rescale_betas_zero_snr=true,
///   timestep_spacing=trailing, steps_offset=1, set_alpha_to_one=false,
///   clip_sample=false, thresholding=false, eta=0（确定性，无 RNG）。
/// 表构造按 torch f32 逐位复刻（linspace 对称分段 FMA；cumprod 内部 double 累乘
/// 逐项转 f32——同 M3 Scheduler.swift init 的本机 arm64 实测结论）；
/// step 标量 Double 计算后转 Float 参与逐元素运算，对标 torch fp32 张量数学。
final class DDIMScheduler {
    let numTrainTimesteps = 1000
    let betaStart = 0.00085
    let betaEnd = 0.012
    let stepsOffset = 1

    /// 1000 个训练期 alphas_cumprod（rescale 后）
    private(set) var alphasCumprod: [Float] = []
    /// set_alpha_to_one=false → alphas_cumprod[0]
    private(set) var finalAlphaCumprod: Float = 0
    private(set) var timesteps: [Int] = []
    private(set) var numInferenceSteps = 0

    init() {
        // betas = linspace(sqrt(start), sqrt(end), 1000)^2（torch f32，对称分段 FMA）
        let s = Float(betaStart.squareRoot()), e = Float(betaEnd.squareRoot())
        let stepF = (e - s) / Float(numTrainTimesteps - 1)
        let half = numTrainTimesteps / 2
        var betas = [Float]()
        betas.reserveCapacity(numTrainTimesteps)
        for i in 0 ..< numTrainTimesteps {
            let v = i < half
                ? s.addingProduct(stepF, Float(i))
                : e.addingProduct(-stepF, Float(numTrainTimesteps - 1 - i))
            betas.append(v * v)  // f32 平方
        }
        betas = DDIMScheduler.rescaleZeroTerminalSNR(betas)
        let alphas = betas.map { Float(1.0) - $0 }
        alphasCumprod = DDIMScheduler.torchCumprodF32(alphas)
        finalAlphaCumprod = alphasCumprod[0]
    }

    /// torch f32 cumprod：本机 arm64 内部 double 累乘、逐项转 f32（M3 实测同款）
    static func torchCumprodF32(_ xs: [Float]) -> [Float] {
        var acc = 1.0
        var out = [Float]()
        out.reserveCapacity(xs.count)
        for x in xs {
            acc *= Double(x)
            out.append(Float(acc))
        }
        return out
    }

    /// rescale_zero_terminal_snr：alphas_bar_sqrt 平移缩放再回推 betas。
    /// 每步都是独立 torch f32 张量运算（-= 与 *= 分开舍入，不做 FMA 合并）。
    static func rescaleZeroTerminalSNR(_ betas: [Float]) -> [Float] {
        let alphas = betas.map { Float(1.0) - $0 }
        let ac = torchCumprodF32(alphas)
        var absSqrt = ac.map { $0.squareRoot() }
        let a0 = absSqrt.first!, aT = absSqrt.last!
        let scale = a0 / (a0 - aT)  // torch f32 标量除
        for i in absSqrt.indices {
            absSqrt[i] -= aT       // Shift so the last timestep is zero
            absSqrt[i] *= scale    // Scale so the first timestep is back
        }
        let bar = absSqrt.map { $0 * $0 }  // **2 revert sqrt
        var alphas2 = [bar[0]]
        for i in 1 ..< bar.count { alphas2.append(bar[i] / bar[i - 1]) }  // revert cumprod
        return alphas2.map { Float(1.0) - $0 }
    }

    /// timestep_spacing="trailing": np.round(np.arange(1000, 0, -1000/n)).astype(int64) - 1
    /// （4 步 → [999,749,499,249]，与锚点 meta 一致）
    func setTimesteps(_ n: Int) {
        numInferenceSteps = n
        let stepRatio = Double(numTrainTimesteps) / Double(n)
        var ts: [Int] = []
        var v = Double(numTrainTimesteps)
        while v > 0 {  // np.arange(1000, 0, -ratio)
            ts.append(Int(v.rounded()) - 1)
            v -= stepRatio
        }
        timesteps = ts
    }

    /// 本 config 下恒等（保持 diffusers 通用接口语义）
    func scaleModelInput(_ sample: [Float], timestep _: Int) -> [Float] { sample }

    /// step(model_output, timestep, sample, eta=0) -> prev_sample，v_prediction 分支:
    ///   x0  = sqrt(alpha_t)*sample - sqrt(beta_t)*model_output
    ///   eps = sqrt(alpha_t)*model_output + sqrt(beta_t)*sample
    ///   prev = sqrt(alpha_t_prev)*x0 + sqrt(1-alpha_t_prev)*eps   （eta=0 无噪声项）
    /// 末步 prev_timestep=-1 → alpha_t_prev = final_alpha_cumprod。
    @discardableResult
    func step(modelOutput: [Float], timestep: Int, sample: [Float]) -> [Float] {
        precondition(numInferenceSteps > 0, "call setTimesteps first")
        precondition(sample.count == modelOutput.count)
        let prevT = timestep - numTrainTimesteps / numInferenceSteps
        let aT = Double(alphasCumprod[timestep])
        let aPrev = prevT >= 0 ? Double(alphasCumprod[prevT]) : Double(finalAlphaCumprod)
        let sqA = Float(aT.squareRoot())
        let sqB = Float((1.0 - aT).squareRoot())
        let sqAP = Float(aPrev.squareRoot())
        let sqBP = Float((1.0 - aPrev).squareRoot())
        var out = [Float](repeating: 0, count: sample.count)
        for i in 0 ..< sample.count {
            let x0 = sqA * sample[i] - sqB * modelOutput[i]
            let eps = sqA * modelOutput[i] + sqB * sample[i]
            out[i] = sqAP * x0 + sqBP * eps
        }
        return out
    }
}
