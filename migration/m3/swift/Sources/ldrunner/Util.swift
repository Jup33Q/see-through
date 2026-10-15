import Foundation

/// 全局路径：以本源码位置推导仓库根，任何 cwd 下都可运行。
enum Paths {
    /// .../migration/m3/swift
    static let packageDir = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()  // ldrunner
        .deletingLastPathComponent()  // Sources
        .deletingLastPathComponent()  // swift
    /// .../migration/m3
    static let m3 = packageDir.deletingLastPathComponent()
    /// 仓库根
    static let repo = m3.deletingLastPathComponent().deletingLastPathComponent()

    static var fixtures: String { packageDir.appendingPathComponent("Fixtures").path }
    static var m3Out: String { m3.appendingPathComponent("out").path }
    static var golden: String { repo.appendingPathComponent("tests/golden/test_image").path }
    static var trajRoot: String { m3.appendingPathComponent("golden_trajectory").path }
    static var unetMerged: String { m3Out + "/models/layerdiff3d_unet_merged_fp16.mlpackage" }
    static var vaeEncoder: String { repo.path + "/migration/m2/out/models/sdxl_vae_encoder_fp16.mlpackage" }
    static var vaeDecoder: String { repo.path + "/migration/m2/out/models/sdxl_vae_decoder_fp16.mlpackage" }
    static var transDecoder: String { repo.path + "/migration/m2/out/models/trans_vae_decoder_fp16.mlpackage" }
    static func statics(_ pass: String) -> String { m3Out + "/statics/\(pass)" }
}

func requireFiles(_ paths: [String], hint: String) throws {
    let missing = paths.filter { !FileManager.default.fileExists(atPath: $0) }
    if !missing.isEmpty {
        throw Err("缺少前置产物（\(hint)）:\n  " + missing.joined(separator: "\n  "))
    }
}

// MARK: - 对比统计

/// dynamic PSNR：peak = max|ref|
func dynPSNR(ref: [Float], out: [Float]) -> (psnr: Double, maxAbs: Double) {
    precondition(ref.count == out.count)
    var mse = 0.0, peak = 0.0, maxAbs = 0.0
    for i in 0 ..< ref.count {
        let d = Double(out[i]) - Double(ref[i])
        mse += d * d
        let a = abs(d)
        if a > maxAbs { maxAbs = a }
        let r = abs(Double(ref[i]))
        if r > peak { peak = r }
    }
    mse /= Double(ref.count)
    let psnr = mse == 0 ? Double.infinity : 10.0 * log10(peak * peak / mse)
    return (psnr, maxAbs)
}

/// uint8 RGBA PSNR（peak=255）
func psnrU8(_ a: [UInt8], _ b: [UInt8]) -> (psnr: Double, maxAbs: Int) {
    precondition(a.count == b.count)
    var mse = 0.0
    var maxAbs = 0
    for i in 0 ..< a.count {
        let d = Int(a[i]) - Int(b[i])
        mse += Double(d * d)
        if abs(d) > maxAbs { maxAbs = abs(d) }
    }
    mse /= Double(a.count)
    return (mse == 0 ? .infinity : 10.0 * log10(255.0 * 255.0 / mse), maxAbs)
}

func fmt(_ v: Double, _ p: Int = 3) -> String { v.isInfinite ? "inf" : String(format: "%.\(p)f", v) }

// MARK: - JSON

func jsonLoad(_ path: String) throws -> [String: Any] {
    let d = try Data(contentsOf: URL(fileURLWithPath: path))
    return try JSONSerialization.jsonObject(with: d) as! [String: Any]
}

func jsonWrite(_ obj: [String: Any], _ path: String?) {
    guard let path else { return }
    var p = path
    if !p.hasPrefix("/") { p = Paths.m3Out + "/" + p }  // 相对路径落在 migration/m3/out/ 下
    try? FileManager.default.createDirectory(atPath: URL(fileURLWithPath: p).deletingLastPathComponent().path, withIntermediateDirectories: true)
    if let d = try? JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted, .sortedKeys]) {
        try? d.write(to: URL(fileURLWithPath: p))
        print("JSON -> \(p)")
    }
}

func nowMs() -> Double { CFAbsoluteTimeGetCurrent() * 1000 }

// MARK: - 锚点 meta.json

struct TrajMeta {
    var frames: Int
    var groupIndex: Int
    var tags: [String]
    var timesteps: [Int]
    var scalingFactor: Double
    var composite: [String: Any]?
    var passName: String { groupIndex == 0 ? "body" : "head" }

    static func load(_ trajDir: String) throws -> TrajMeta {
        let m = try jsonLoad(trajDir + "/meta.json")
        return TrajMeta(
            frames: (m["frames"] as! NSNumber).intValue,
            groupIndex: (m["group_index"] as! NSNumber).intValue,
            tags: m["tags"] as! [String],
            timesteps: (m["timesteps"] as! [Any]).map { Int(($0 as! NSNumber).doubleValue.rounded()) },
            scalingFactor: (m["scaling_factor"] as! NSNumber).doubleValue,
            composite: m["composite"] as? [String: Any]
        )
    }
}

// MARK: - 帧展开工具（(1,4,160,160) → (1,F,4,160,160) 等）

/// 把单帧 (C,H,W) 沿帧维复制 F 份（对标 torch expand）
func expandFrames(_ frame: [Float], frames: Int) -> [Float] {
    var out = [Float]()
    out.reserveCapacity(frame.count * frames)
    for _ in 0 ..< frames { out.append(contentsOf: frame) }
    return out
}
