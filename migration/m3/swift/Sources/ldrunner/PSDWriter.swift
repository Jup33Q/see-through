import Foundation

private struct BEWriter {
    private(set) var data = Data()

    var count: Int { data.count }

    mutating func u8(_ value: UInt8) { data.append(value) }
    mutating func u16(_ value: UInt16) {
        u8(UInt8(value >> 8)); u8(UInt8(value & 0xff))
    }
    mutating func i16(_ value: Int16) { u16(UInt16(bitPattern: value)) }
    mutating func u32(_ value: UInt32) {
        u8(UInt8(value >> 24)); u8(UInt8((value >> 16) & 0xff))
        u8(UInt8((value >> 8) & 0xff)); u8(UInt8(value & 0xff))
    }
    mutating func i32(_ value: Int32) { u32(UInt32(bitPattern: value)) }
    mutating func bytes(_ bytes: [UInt8]) { data.append(contentsOf: bytes) }
    mutating func bytes(_ bytes: Data) { data.append(bytes) }
    mutating func ascii(_ value: String) { bytes(Array(value.utf8)) }

    mutating func pad(to alignment: Int) {
        while data.count % alignment != 0 { u8(0) }
    }
}

private func packBitsRow(_ plane: [UInt8], start: Int, width: Int) -> [UInt8] {
    var out: [UInt8] = []
    out.reserveCapacity(width + width / 128 + 1)
    var i = 0
    while i < width {
        var run = 1
        while i + run < width && run < 128 && plane[start + i + run] == plane[start + i] { run += 1 }
        if run >= 3 {
            out.append(UInt8(257 - run))
            out.append(plane[start + i])
            i += run
            continue
        }
        let literalStart = i
        i += 1
        while i < width && i - literalStart < 128 {
            if i + 2 < width && plane[start + i] == plane[start + i + 1]
                && plane[start + i] == plane[start + i + 2] { break }
            i += 1
        }
        out.append(UInt8(i - literalStart - 1))
        out.append(contentsOf: plane[start + literalStart ..< start + i])
    }
    return out
}

private func encodeRLEPlanes(_ planes: [[UInt8]], w: Int, h: Int) throws -> Data {
    var compressedRows: [[UInt8]] = []
    compressedRows.reserveCapacity(planes.count * h)
    for plane in planes {
        guard plane.count == w * h else { throw Err("PSD channel size mismatch") }
        for y in 0 ..< h {
            let row = packBitsRow(plane, start: y * w, width: w)
            guard row.count <= Int(UInt16.max) else { throw Err("PSD RLE row too long") }
            compressedRows.append(row)
        }
    }
    var out = BEWriter()
    for row in compressedRows { out.u16(UInt16(row.count)) }
    for row in compressedRows { out.bytes(row) }
    return out.data
}

private func planes(for part: PostPart) -> [[UInt8]] {
    let count = part.image.w * part.image.h
    var alpha = [UInt8](repeating: 0, count: count)
    var red = [UInt8](repeating: 0, count: count)
    var green = [UInt8](repeating: 0, count: count)
    var blue = [UInt8](repeating: 0, count: count)
    for i in 0 ..< count {
        red[i] = part.image.px[i * 4]
        green[i] = part.image.px[i * 4 + 1]
        blue[i] = part.image.px[i * 4 + 2]
        alpha[i] = part.image.px[i * 4 + 3]
    }
    return [alpha, red, green, blue, alpha]
}

private func compositePreview(_ layers: [PostPart], w: Int, h: Int) -> [UInt8] {
    var color = [Float](repeating: 0, count: w * h * 3)
    var coverage = [Float](repeating: 0, count: w * h)
    for part in layers {
        let r = part.rect
        for y in 0 ..< part.image.h {
            let cy = r.y1 + y
            if cy < 0 || cy >= h { continue }
            for x in 0 ..< part.image.w {
                let cx = r.x1 + x
                if cx < 0 || cx >= w { continue }
                let si = y * part.image.w + x
                let di = cy * w + cx
                let a = Float(part.image.px[si * 4 + 3]) / Float(255)
                let shape = a * a
                coverage[di] = coverage[di] + shape - coverage[di] * shape
                for c in 0 ..< 3 {
                    let source = Float(part.image.px[si * 4 + c]) / Float(255)
                    color[di * 3 + c] = (Float(1) - shape) * color[di * 3 + c] + shape * source
                }
            }
        }
    }
    var rgba = [UInt8](repeating: 255, count: w * h * 4)
    for i in 0 ..< w * h {
        let divisor = coverage[i]
        for c in 0 ..< 3 {
            var value = divisor > 0 ? color[i * 3 + c] / divisor : 0
            value = min(1, max(0, value))
            rgba[i * 4 + c] = UInt8(min(255, max(0, Int(value * Float(255)))))
        }
    }
    return rgba
}

private func interleavedToPlanes(_ rgba: [UInt8], count: Int) -> [[UInt8]] {
    var planes = [[UInt8](repeating: 0, count: count),
                  [UInt8](repeating: 0, count: count),
                  [UInt8](repeating: 0, count: count),
                  [UInt8](repeating: 0, count: count)]
    for i in 0 ..< count {
        for c in 0 ..< 4 { planes[c][i] = rgba[i * 4 + c] }
    }
    return planes
}

enum PSDWriter {
    static func write(layers: [PostPart], canvasW: Int, canvasH: Int, path: String) throws {
        guard canvasW > 0 && canvasH > 0 && canvasW <= 30000 && canvasH <= 30000 else {
            throw Err("invalid PSD canvas \(canvasW)x\(canvasH)")
        }
        var channelData: [[Data]] = []
        channelData.reserveCapacity(layers.count)
        for part in layers {
            guard part.image.w == part.rect.width && part.image.h == part.rect.height else {
                throw Err("layer rect/image mismatch for \(part.tag)")
            }
            var encoded: [Data] = []
            for plane in planes(for: part) {
                var channel = BEWriter()
                channel.u16(1)
                channel.bytes(try encodeRLEPlanes([plane], w: part.image.w, h: part.image.h))
                encoded.append(channel.data)
            }
            channelData.append(encoded)
        }

        var records = BEWriter()
        for (index, part) in layers.enumerated() {
            records.i32(Int32(part.rect.y1)); records.i32(Int32(part.rect.x1))
            records.i32(Int32(part.rect.y2)); records.i32(Int32(part.rect.x2))
            records.u16(5)
            for (channelIndex, channelID) in [Int16(-1), 0, 1, 2, -2].enumerated() {
                records.i16(channelID)
                records.u32(UInt32(channelData[index][channelIndex].count))
            }
            records.ascii("8BIM"); records.ascii("norm")
            records.u8(255); records.u8(0); records.u8(8); records.u8(0)

            var extra = BEWriter()
            extra.u32(20)
            extra.i32(Int32(part.rect.y1)); extra.i32(Int32(part.rect.x1))
            extra.i32(Int32(part.rect.y2)); extra.i32(Int32(part.rect.x2))
            extra.u8(0); extra.u8(0); extra.u16(0)
            extra.u32(40)
            for _ in 0 ..< 5 {
                extra.u16(0); extra.u16(65535); extra.u16(0); extra.u16(65535)
            }
            guard let name = part.tag.data(using: .macOSRoman), !name.isEmpty, name.count <= 255 else {
                throw Err("invalid PSD layer name: \(part.tag)")
            }
            extra.u8(UInt8(name.count)); extra.bytes(name)
            extra.pad(to: 4)
            extra.pad(to: 2)
            records.u32(UInt32(extra.count)); records.bytes(extra.data)
        }

        var layerInfo = BEWriter()
        layerInfo.i16(Int16(layers.count))
        layerInfo.bytes(records.data)
        for encoded in channelData {
            for channel in encoded { layerInfo.bytes(channel) }
        }
        layerInfo.pad(to: 4)

        var layerAndMask = BEWriter()
        layerAndMask.u32(UInt32(layerInfo.count)); layerAndMask.bytes(layerInfo.data)
        layerAndMask.u32(0)

        var document = BEWriter()
        document.ascii("8BPS"); document.u16(1)
        document.bytes([0, 0, 0, 0, 0, 0])
        document.u16(4); document.u32(UInt32(canvasH)); document.u32(UInt32(canvasW))
        document.u16(8); document.u16(3)
        document.u32(0); document.u32(0)
        document.u32(UInt32(layerAndMask.count)); document.bytes(layerAndMask.data)

        let preview = compositePreview(layers, w: canvasW, h: canvasH)
        document.u16(1)
        document.bytes(try encodeRLEPlanes(interleavedToPlanes(preview, count: canvasW * canvasH),
                                             w: canvasW, h: canvasH))
        let url = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(atPath: url.deletingLastPathComponent().path,
                                                withIntermediateDirectories: true)
        try document.data.write(to: url, options: .atomic)
    }
}
