import CoreML
import Foundation

/// CoreML 封装：fp32 MLMultiArray IO（CF-contiguous，直接 dataPointer 读写）。
final class CoreMLModel {
    let model: MLModel
    let path: String

    init(path: String) throws {
        self.path = path
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all
        let t0 = nowMs()
        // macOS 新版要求 .mlpackage 先编译为 .mlmodelc
        let compiled = try MLModel.compileModel(at: URL(fileURLWithPath: path))
        model = try MLModel(contentsOf: compiled, configuration: cfg)
        print("[coreml] loaded \(URL(fileURLWithPath: path).lastPathComponent) in \(fmt(nowMs() - t0, 0))ms")
    }

    /// 同步推理，返回首个输出（或指定名）的 [Float] + 形状
    @discardableResult
    func predict(_ inputs: [String: MLMultiArray], output name: String? = nil) throws -> (data: [Float], shape: [Int]) {
        let provider = try MLDictionaryFeatureProvider(dictionary: inputs.mapValues { MLFeatureValue(multiArray: $0) })
        let out = try model.prediction(from: provider)
        let key = name ?? out.featureNames.first!
        guard let ma = out.featureValue(for: key)?.multiArrayValue else {
            throw Err("output \(key) not a multiarray (available: \(out.featureNames))")
        }
        return (ma.toFloatArray(), ma.shape.map { $0.intValue })
    }
}

func makeMA(_ shape: [Int]) throws -> MLMultiArray {
    try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
}

extension MLMultiArray {
    /// C-contiguous → [Float]（fp32 直通；fp16 逐元素转换——M4 VAE decoder 输出为 fp16）
    func toFloatArray() -> [Float] {
        let n = count
        switch dataType {
        case .float32:
            let ptr = dataPointer.bindMemory(to: Float.self, capacity: n)
            return Array(UnsafeBufferPointer(start: ptr, count: n))
        case .float16:
            let ptr = dataPointer.bindMemory(to: Float16.self, capacity: n)
            return UnsafeBufferPointer(start: ptr, count: n).map { Float($0) }
        default:
            preconditionFailure("toFloatArray: unsupported dtype \(dataType.rawValue)")
        }
    }

    /// 写入 C-contiguous fp32
    func fill(_ data: [Float]) {
        precondition(data.count == count, "fill: \(data.count) != MA count \(count)")
        let ptr = dataPointer.bindMemory(to: Float.self, capacity: data.count)
        data.withUnsafeBytes { src in
            UnsafeMutableRawBufferPointer(start: ptr, count: data.count * 4).copyMemory(from: src)
        }
    }

    convenience init(filledWith data: [Float], shape: [Int]) throws {
        try self.init(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        fill(data)
    }
}

// MARK: - 模型加载便捷入口（带"待模型"提示）

func loadUNet() throws -> CoreMLModel {
    try requireFiles([Paths.unetMerged], hint: "merged UNet 转换脚本尚未完成 → 待模型")
    return try CoreMLModel(path: Paths.unetMerged)
}

struct Statics {
    var encoderHiddenStates: [Float]  // (F,77,2048)
    var textEmbeds: [Float]           // (F,1280)
    var timeIds: [Float]              // (F,6)
    var ehsShape: [Int]
    var teShape: [Int]
    var tiShape: [Int]

    static func load(_ pass: String) throws -> Statics {
        let dir = Paths.statics(pass)
        try requireFiles(["\(dir)/encoder_hidden_states.npy", "\(dir)/text_embeds.npy", "\(dir)/time_ids.npy"],
                         hint: "statics 尚未生成（02_convert_unet_merged.py 前置产物）→ 待模型")
        let ehs = try Npy.load("\(dir)/encoder_hidden_states.npy")
        let te = try Npy.load("\(dir)/text_embeds.npy")
        let ti = try Npy.load("\(dir)/time_ids.npy")
        return Statics(encoderHiddenStates: ehs.data, textEmbeds: te.data, timeIds: ti.data,
                       ehsShape: ehs.shape, teShape: te.shape, tiShape: ti.shape)
    }
}

/// UNet 加载：优先 merged 包（条件 = out/statics/<pass>/ 预增强 embeds）；
/// merged 缺失时回退 M1 spike-B 单帧数包（IO 契约相同，group 增强在包内，
/// 条件 = 锚点原始 prompt_embeds/pooled_embeds/time_ids）。
struct UNetBundle {
    var model: CoreMLModel
    var cond: Statics
    var flavor: String
}

func loadUNetBundle(pass: String, traj: String) throws -> UNetBundle {
    if FileManager.default.fileExists(atPath: Paths.unetMerged) {
        return try UNetBundle(model: loadUNet(), cond: Statics.load(pass), flavor: "merged")
    }
    let tag = pass == "body" ? "body13_160" : "head11_160"
    let spike = Paths.repo.path + "/spikes/m1/spike_b_unet/unet_\(tag).mlpackage"
    try requireFiles([spike, "\(traj)/prompt_embeds.npy", "\(traj)/pooled_embeds.npy"],
                     hint: "merged UNet 与 spike-B 回退包均不可用 → 待模型")
    let ehs = try Npy.load("\(traj)/prompt_embeds.npy")
    let te = try Npy.load("\(traj)/pooled_embeds.npy")
    // 锚点未存 time_ids；其值恒定 [1280,1280,0,0,1280,1280] 逐帧重复（diffusers_kdiffusion_sdxl.py）
    let f = ehs.shape[0]
    let oneTid: [Float] = [1280, 1280, 0, 0, 1280, 1280]
    let ti = (0 ..< f).flatMap { _ in oneTid }
    let cond = Statics(encoderHiddenStates: ehs.data, textEmbeds: te.data, timeIds: ti,
                       ehsShape: ehs.shape, teShape: te.shape, tiShape: [f, 6])
    print("[unet] merged 包缺失，回退 spike-B \(tag)（条件=锚点原始 embeds）")
    return try UNetBundle(model: CoreMLModel(path: spike), cond: cond, flavor: "spikeb")
}
