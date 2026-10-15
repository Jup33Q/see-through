import Foundation

/// DPMSolverMultistepScheduler 的 Swift 精确移植。
/// 权威来源: diffusers 0.37.0 scheduling_dpmsolver_multistep.py。
/// 固定 config（layerdiff3d scheduler_config.json）:
///   algorithm_type=sde-dpmsolver++, prediction_type=epsilon, solver_order=2,
///   solver_type=midpoint, lower_order_final=true, final_sigmas_type=zero,
///   timestep_spacing=leading, steps_offset=1, num_train_timesteps=1000,
///   beta_schedule=scaled_linear, beta_start=0.00085, beta_end=0.012,
///   skip_prk_steps=true, use_karras_sigmas=false。
/// 标量用 Double、张量逐元素 Float（对标 torch float32 张量数学）。
final class DPMSolver {
    let numTrainTimesteps = 1000
    let betaStart = 0.00085
    let betaEnd = 0.012
    let solverOrder = 2
    let stepsOffset = 1
    let initNoiseSigma: Double = 1.0

    /// 1000 个训练期 sigma，按 torch float32 运算逐步复刻
    private(set) var trainSigmas: [Float] = []
    /// set_timesteps 后: numSteps+1 个（末尾补 0，final_sigmas_type="zero"）
    private(set) var sigmas: [Double] = []
    private(set) var timesteps: [Int] = []

    private var modelOutputs: [[Float]?]
    private var lowerOrderNums = 0
    private(set) var stepIndex: Int? = nil
    /// 上一步 step 的噪声系数 sigma_t*sqrt(1-exp(-2h))（sched-check 反推用）
    private(set) var lastNoiseScale: Double = 0
    private(set) var lastOrder: Int = 0

    init() {
        // 逐位复刻 torch（本机 arm64 实测）:
        //   linspace: f32 端点，step=Float((e-s)/999)，i<500 时 fma(step,i,start)，
        //             否则 fma(-step,999-i,end)（对称分段 FMA，单舍入）
        //   betas = linspace^2（f32 乘）；alphas_cumprod 内部以 double 累乘再逐项转 f32
        let s = Float(betaStart.squareRoot()), e = Float(betaEnd.squareRoot())
        let step = (e - s) / Float(numTrainTimesteps - 1)
        let half = numTrainTimesteps / 2
        var acD = 1.0
        trainSigmas.reserveCapacity(numTrainTimesteps)
        for i in 0 ..< numTrainTimesteps {
            let v = i < half
                ? s.addingProduct(step, Float(i))
                : e.addingProduct(-step, Float(numTrainTimesteps - 1 - i))
            let beta = v * v                          // f32 平方
            acD *= Double(1.0 - beta)                 // double 累乘
            let ac = Float(acD)                       // 逐项转 f32
            trainSigmas.append(((1.0 - ac) / ac).squareRoot())  // f32
        }
        modelOutputs = Array(repeating: nil, count: solverOrder)
    }

    /// timestep_spacing="leading": step_ratio = 1000 // (n+1)；
    /// timesteps = (arange(n+1)*ratio)[::-1][:-1] + steps_offset
    func setTimesteps(_ n: Int) {
        let stepRatio = numTrainTimesteps / (n + 1)
        var ts: [Int] = []
        for i in 0 ..< n { ts.append((n - i) * stepRatio + stepsOffset) }
        timesteps = ts
        // np.interp 在整数点上恒等 → 直接索引；末尾拼 sigma_last=0
        sigmas = ts.map { Double(trainSigmas[$0]) } + [0.0]
        modelOutputs = Array(repeating: nil, count: solverOrder)
        lowerOrderNums = 0
        stepIndex = nil
    }

    /// 本 config 下恒等（保持 diffusers 通用接口语义）
    func scaleModelInput(_ sample: [Float], timestep _: Int) -> [Float] { sample }

    /// _sigma_to_alpha_sigma_t: alpha_t = 1/sqrt(s^2+1); sigma_t = s*alpha_t
    private func alphaSigmaT(_ sigma: Double) -> (alpha: Double, sigma: Double) {
        let a = 1.0 / (sigma * sigma + 1.0).squareRoot()
        return (a, sigma * a)
    }

    /// convert_model_output (sde-dpmsolver++, epsilon): x0 = (sample - sigma_t*eps)/alpha_t
    private func convertModelOutput(_ modelOutput: [Float], sample: [Float]) -> [Float] {
        let sigma = sigmas[stepIndex!]
        let (alphaT, sigmaT) = alphaSigmaT(sigma)
        let st = Float(sigmaT), invA = Float(1.0 / alphaT)
        var out = [Float](repeating: 0, count: sample.count)
        for i in 0 ..< sample.count {
            out[i] = (sample[i] - st * modelOutput[i]) * invA
        }
        return out
    }

    /// dpm_solver_first_order_update, sde-dpmsolver++ 分支:
    /// x_t = (sigma_t/sigma_s * e^-h)*sample + alpha_t*(1-e^-2h)*D0 + sigma_t*sqrt(1-e^-2h)*noise
    private func firstOrderUpdate(_ d0: [Float], sample: [Float], noise: [Float]) -> [Float] {
        let i = stepIndex!
        let (alphaT, sigmaT) = alphaSigmaT(sigmas[i + 1])
        let (alphaS, sigmaS) = alphaSigmaT(sigmas[i])
        let h = (log(alphaT) - log(sigmaT)) - (log(alphaS) - log(sigmaS))
        let cSample = sigmaT / sigmaS * exp(-h)
        let cD0 = alphaT * (1.0 - exp(-2.0 * h))
        let k = sigmaT * (1.0 - exp(-2.0 * h)).squareRoot()
        lastNoiseScale = k
        let cs = Float(cSample), cd = Float(cD0), ck = Float(k)
        var out = [Float](repeating: 0, count: sample.count)
        for j in 0 ..< sample.count {
            out[j] = cs * sample[j] + cd * d0[j] + ck * noise[j]
        }
        return out
    }

    /// multistep_dpm_solver_second_order_update, sde-dpmsolver++ + midpoint:
    /// D0 = m0; D1 = (m0-m1)/r0; r0 = h_0/h
    /// x_t = (sigma_t/sigma_s0 * e^-h)*sample + alpha_t*(1-e^-2h)*(D0 + 0.5*D1) + sigma_t*sqrt(1-e^-2h)*noise
    private func secondOrderUpdate(_ m0: [Float], _ m1: [Float], sample: [Float], noise: [Float]) -> [Float] {
        let i = stepIndex!
        let (alphaT, sigmaT) = alphaSigmaT(sigmas[i + 1])
        let (alphaS0, sigmaS0) = alphaSigmaT(sigmas[i])
        let (alphaS1, sigmaS1) = alphaSigmaT(sigmas[i - 1])
        let lambdaT = log(alphaT) - log(sigmaT)
        let h = lambdaT - (log(alphaS0) - log(sigmaS0))
        let h0 = (log(alphaS0) - log(sigmaS0)) - (log(alphaS1) - log(sigmaS1))
        let r0 = h0 / h
        let cSample = sigmaT / sigmaS0 * exp(-h)
        let cD0 = alphaT * (1.0 - exp(-2.0 * h))
        let k = sigmaT * (1.0 - exp(-2.0 * h)).squareRoot()
        lastNoiseScale = k
        let cs = Float(cSample), cd = Float(cD0), cd1 = Float(0.5 * cD0), ck = Float(k)
        let invR = Float(1.0 / r0)
        var out = [Float](repeating: 0, count: sample.count)
        for j in 0 ..< sample.count {
            let d1 = invR * (m0[j] - m1[j])
            out[j] = cs * sample[j] + cd * m0[j] + cd1 * d1 + ck * noise[j]
        }
        return out
    }

    /// step(model_output, timestep, sample, variance_noise) -> prev_sample
    /// diffusers 里 sde 变体每步内部 randn；Swift 侧由调用方显式给 varianceNoise
    ///（nil → 全零噪声，用于确定性/反推模式）。
    @discardableResult
    func step(modelOutput: [Float], timestep: Int, sample: [Float], varianceNoise: [Float]? = nil) -> [Float] {
        precondition(!timesteps.isEmpty, "call setTimesteps first")
        if stepIndex == nil {
            stepIndex = timesteps.firstIndex(of: timestep) ?? timesteps.count - 1
        }
        let i = stepIndex!
        // final_sigmas_type="zero" ⇒ 末步 lower_order_final 必真
        let lowerOrderFinal = i == timesteps.count - 1

        let x0 = convertModelOutput(modelOutput, sample: sample)
        for k in 0 ..< solverOrder - 1 { modelOutputs[k] = modelOutputs[k + 1] }
        modelOutputs[solverOrder - 1] = x0

        let noise = varianceNoise ?? [Float](repeating: 0, count: sample.count)

        let prev: [Float]
        if lowerOrderNums < 1 || lowerOrderFinal {
            prev = firstOrderUpdate(x0, sample: sample, noise: noise)
            lastOrder = 1
        } else {
            prev = secondOrderUpdate(modelOutputs[1]!, modelOutputs[0]!, sample: sample, noise: noise)
            lastOrder = 2
        }
        if lowerOrderNums < solverOrder { lowerOrderNums += 1 }
        stepIndex = i + 1
        return prev
    }
}
