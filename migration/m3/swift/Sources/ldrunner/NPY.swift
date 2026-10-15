import Foundation

/// numpy .npy v1/v2 读取（'<f4' C-order → shape + [Float]）。
/// 锚点文件全是 fp32 little-endian，arm64 与之一致。
struct Npy {
    var shape: [Int]
    var data: [Float]

    var count: Int { data.count }

    static func load(_ path: String) throws -> Npy {
        let url = URL(fileURLWithPath: path)
        let raw = try Data(contentsOf: url)
        guard raw.count > 12 else { throw Err("npy too small: \(path)") }
        let b = [UInt8](raw)
        guard b[0] == 0x93, b[1] == 0x4E, b[2] == 0x55, b[3] == 0x4D,
              b[4] == 0x50, b[5] == 0x59 else { throw Err("bad npy magic: \(path)") }
        let major = b[6]
        let headerLen: Int
        let dataStart: Int
        if major == 1 {
            headerLen = Int(b[8]) | Int(b[9]) << 8
            dataStart = 10 + headerLen
        } else {
            headerLen = Int(b[8]) | Int(b[9]) << 8 | Int(b[10]) << 16 | Int(b[11]) << 24
            dataStart = 12 + headerLen
        }
        guard dataStart <= raw.count else { throw Err("bad npy header len: \(path)") }
        let header = String(decoding: b[10 + (major == 1 ? 0 : 2) ..< 10 + (major == 1 ? 0 : 2) + headerLen], as: UTF8.self)
        guard header.contains("'descr': '<f4'") || header.contains("\"descr\": \"<f4\"")
        else { throw Err("npy descr not '<f4': \(path) header=\(header)") }
        guard !header.contains("'fortran_order': True") else { throw Err("fortran order unsupported: \(path)") }
        let shape = try parseShape(header, path)
        let n = shape.reduce(1, *)
        guard dataStart + n * 4 <= raw.count else { throw Err("npy data truncated: \(path)") }
        var out = [Float](repeating: 0, count: n)
        out.withUnsafeMutableBytes { dst in
            raw.copyBytes(to: dst, from: dataStart ..< dataStart + n * 4)
        }
        return Npy(shape: shape, data: out)
    }

    /// 从 "'shape': (1, 13, 8, 160, 160)," 中提取维度
    private static func parseShape(_ header: String, _ path: String) throws -> [Int] {
        guard let keyRange = header.range(of: "'shape':") ?? header.range(of: "\"shape\":") else {
            throw Err("npy header has no shape: \(path)")
        }
        let rest = header[keyRange.upperBound...]
        guard let open = rest.firstIndex(of: "("), let close = rest.firstIndex(of: ")"), open < close else {
            throw Err("npy bad shape tuple: \(path)")
        }
        let inside = rest[rest.index(after: open) ..< close]
        let dims = inside.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        guard !dims.isEmpty else { throw Err("npy empty shape: \(path)") }
        return dims
    }

    /// 写 v1.0（调试/中间结果用）
    func save(_ path: String) throws {
        let shapeStr: String
        if shape.count == 1 { shapeStr = "(\(shape[0]),)" } else { shapeStr = "(" + shape.map(String.init).joined(separator: ", ") + ")" }
        var header = "{'descr': '<f4', 'fortran_order': False, 'shape': \(shapeStr), }"
        // 补齐到 64 字节对齐（含 magic 10 字节与结尾换行）
        let base = 10 + header.count + 1
        let pad = (64 - base % 64) % 64
        header += String(repeating: " ", count: pad) + "\n"
        var out = Data()
        out.append(contentsOf: [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 1, 0])
        let hl = UInt16(header.count)
        out.append(UInt8(hl & 0xFF)); out.append(UInt8(hl >> 8))
        out.append(contentsOf: header.utf8)
        data.withUnsafeBytes { out.append(contentsOf: $0) }
        try out.write(to: URL(fileURLWithPath: path))
    }
}

struct Err: Error, CustomStringConvertible {
    let description: String
    init(_ s: String) { description = s }
}
