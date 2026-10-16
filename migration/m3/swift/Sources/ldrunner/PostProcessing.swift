import CoreGraphics
import Foundation
import ImageIO

struct GrayImage {
    var w: Int
    var h: Int
    var px: [UInt8]

    static func load(_ path: String) throws -> GrayImage {
        let url = URL(fileURLWithPath: path) as CFURL
        guard let src = CGImageSourceCreateWithURL(url, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { throw Err("cannot read depth png: \(path)") }
        let w = img.width, h = img.height
        var buf = [UInt8](repeating: 0, count: w * h)
        guard let ctx = CGContext(data: &buf, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: w, space: CGColorSpaceCreateDeviceGray(),
                                  bitmapInfo: 0)
        else { throw Err("cannot create gray context for \(path)") }
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
        return GrayImage(w: w, h: h, px: buf)
    }

    func cropped(_ r: PartRect) -> GrayImage {
        var out = [UInt8](repeating: 0, count: r.width * r.height)
        for y in 0 ..< r.height {
            let src = (r.y1 + y) * w + r.x1
            let dst = y * r.width
            out.replaceSubrange(dst ..< dst + r.width, with: px[src ..< src + r.width])
        }
        return GrayImage(w: r.width, h: r.height, px: out)
    }
}

struct PartRect: Equatable {
    var x1: Int
    var y1: Int
    var x2: Int
    var y2: Int
    var width: Int { x2 - x1 }
    var height: Int { y2 - y1 }
    var array: [Int] { [x1, y1, x2, y2] }
}

final class PostPart {
    var tag: String
    var image: RGBAImage
    var depth: [Float]
    var rect: PartRect
    var partID: Int?
    var depthMedian: Float = 1

    init(tag: String, image: RGBAImage, depth: [Float], rect: PartRect, partID: Int? = nil) {
        self.tag = tag
        self.image = image
        self.depth = depth
        self.rect = rect
        self.partID = partID
    }
}

struct OrderedParts {
    private(set) var keys: [String] = []
    private var storage: [String: PostPart] = [:]

    var values: [PostPart] { keys.map { storage[$0]! } }

    subscript(_ tag: String) -> PostPart? { storage[tag] }

    mutating func insert(_ part: PostPart, as tag: String? = nil) {
        let key = tag ?? part.tag
        if storage[key] == nil { keys.append(key) }
        part.tag = key
        storage[key] = part
    }

    mutating func remove(_ tag: String) -> PostPart? {
        guard let part = storage.removeValue(forKey: tag) else { return nil }
        keys.removeAll { $0 == tag }
        return part
    }
}

struct ComponentStats {
    var left: Int
    var top: Int
    var width: Int
    var height: Int
    var area: Int
}

struct ConnectedComponents {
    var labels: [Int]
    var stats: [ComponentStats]
    var w: Int
    var h: Int
}

private func alphaMask(_ image: RGBAImage) -> [UInt8] {
    var mask = [UInt8](repeating: 0, count: image.w * image.h)
    for i in 0 ..< mask.count { mask[i] = image.px[i * 4 + 3] > 10 ? 255 : 0 }
    return mask
}

private func boundingRect(_ mask: [UInt8], w: Int, h: Int) -> PartRect? {
    var x1 = w, y1 = h, x2 = -1, y2 = -1
    for y in 0 ..< h {
        for x in 0 ..< w where mask[y * w + x] != 0 {
            x1 = min(x1, x); y1 = min(y1, y)
            x2 = max(x2, x); y2 = max(y2, y)
        }
    }
    return x2 < x1 ? nil : PartRect(x1: x1, y1: y1, x2: x2 + 1, y2: y2 + 1)
}

private func cropValues<T>(_ values: [T], w: Int, h: Int, r: PartRect) -> [T] {
    precondition(r.x1 >= 0 && r.y1 >= 0 && r.x2 <= w && r.y2 <= h && r.width >= 0 && r.height >= 0)
    var out = [T]()
    out.reserveCapacity(r.width * r.height)
    for y in r.y1 ..< r.y2 {
        let start = y * w + r.x1
        out.append(contentsOf: values[start ..< start + r.width])
    }
    return out
}

private extension RGBAImage {
    func cropped(_ r: PartRect) -> RGBAImage {
        var out = [UInt8](repeating: 0, count: r.width * r.height * 4)
        for y in 0 ..< r.height {
            let src = ((r.y1 + y) * w + r.x1) * 4
            let dst = y * r.width * 4
            out.replaceSubrange(dst ..< dst + r.width * 4, with: px[src ..< src + r.width * 4])
        }
        return RGBAImage(w: r.width, h: r.height, px: out)
    }
}

private func medianF32(_ values: [Float]) -> Float {
    precondition(!values.isEmpty)
    let sorted = values.sorted()
    let n = sorted.count
    if n % 2 == 1 { return sorted[n / 2] }
    return (sorted[n / 2 - 1] + sorted[n / 2]) / Float(2)
}

private func connectedComponents(_ mask: [UInt8], w: Int, h: Int) -> ConnectedComponents {
    var labels = [Int](repeating: -1, count: w * h)
    var stats = [ComponentStats(left: 0, top: 0, width: w, height: h, area: 0)]
    var foreground = 0
    let dx = [-1, 0, 1, -1, 1, -1, 0, 1]
    let dy = [-1, -1, -1, 0, 0, 1, 1, 1]

    for y in 0 ..< h {
        for x in 0 ..< w {
            let start = y * w + x
            if mask[start] == 0 || labels[start] >= 0 { continue }
            let id = stats.count
            var stack = [start]
            labels[start] = id
            var area = 0, x1 = x, y1 = y, x2 = x, y2 = y
            while let idx = stack.popLast() {
                let cy = idx / w, cx = idx - cy * w
                area += 1
                x1 = min(x1, cx); y1 = min(y1, cy)
                x2 = max(x2, cx); y2 = max(y2, cy)
                for k in 0 ..< 8 {
                    let nx = cx + dx[k], ny = cy + dy[k]
                    if nx < 0 || ny < 0 || nx >= w || ny >= h { continue }
                    let ni = ny * w + nx
                    if mask[ni] != 0 && labels[ni] < 0 {
                        labels[ni] = id
                        stack.append(ni)
                    }
                }
            }
            foreground += area
            stats.append(ComponentStats(left: x1, top: y1, width: x2 - x1 + 1,
                                        height: y2 - y1 + 1, area: area))
        }
    }
    stats[0].area = w * h - foreground
    return ConnectedComponents(labels: labels, stats: stats, w: w, h: h)
}

private func statsOrderDescending(_ stats: [ComponentStats]) -> [Int] {
    let ordered = Array(stats.indices).sorted { a, b in
        if stats[a].area != stats[b].area { return stats[a].area < stats[b].area }
        return a < b
    }
    return Array(ordered.reversed().dropFirst())
}

private func labelMask(_ cc: ConnectedComponents, id: Int) -> [UInt8] {
    cc.labels.map { $0 == id ? 255 : 0 }
}

private func labelLRSplit(_ cc: ConnectedComponents, _ id1: Int, _ id2: Int)
    -> (mask1: [UInt8], mask2: [UInt8], stats1: ComponentStats, stats2: ComponentStats) {
    let s1 = cc.stats[id1], s2 = cc.stats[id2]
    let x1 = Double(s1.left) + Double(s1.width) / 2
    let x2 = Double(s2.left) + Double(s2.width) / 2
    if x2 < x1 {
        return (labelMask(cc, id: id2), labelMask(cc, id: id1), s2, s1)
    }
    return (labelMask(cc, id: id1), labelMask(cc, id: id2), s1, s2)
}

private func dilateCross(_ mask: [UInt8], w: Int, h: Int) -> [UInt8] {
    var out = [UInt8](repeating: 0, count: w * h)
    let dx = [0, -1, 1, 0, 0], dy = [0, 0, 0, -1, 1]
    for y in 0 ..< h {
        for x in 0 ..< w where mask[y * w + x] != 0 {
            for k in 0 ..< 5 {
                let nx = x + dx[k], ny = y + dy[k]
                if nx >= 0 && ny >= 0 && nx < w && ny < h { out[ny * w + nx] = 1 }
            }
        }
    }
    return out
}

private func processCuts(_ part: PostPart, stats: ComponentStats, mask: [UInt8]? = nil) -> PostPart {
    let local = PartRect(x1: stats.left, y1: stats.top,
                         x2: stats.left + stats.width, y2: stats.top + stats.height)
    var image = part.image.cropped(local)
    var depth = cropValues(part.depth, w: part.image.w, h: part.image.h, r: local)
    if let mask {
        let binary = cropValues(mask, w: part.image.w, h: part.image.h, r: local).map { $0 > 15 ? UInt8(1) : 0 }
        let dilated = dilateCross(binary, w: local.width, h: local.height)
        for i in 0 ..< dilated.count {
            image.px[i * 4 + 3] *= dilated[i]
            depth[i] = Float(1) - (Float(1) - depth[i]) * Float(dilated[i])
        }
    }
    let global = PartRect(x1: part.rect.x1 + local.x1, y1: part.rect.y1 + local.y1,
                          x2: part.rect.x1 + local.x2, y2: part.rect.y1 + local.y2)
    return PostPart(tag: part.tag, image: image, depth: depth, rect: global)
}

private func loadPart(src: String, tag: String, minWidth: Int = 64, minSize: Int = 12) throws -> PostPart? {
    let image = try RGBAImage.load(src + "/" + tag + ".png")
    let depthImage = try GrayImage.load(src + "/" + tag + "_depth.png")
    guard image.w == depthImage.w && image.h == depthImage.h else {
        throw Err("image/depth size mismatch for \(tag)")
    }
    let w = image.w, h = image.h
    let mask = alphaMask(image)
    let pTest = max(w, h) / 10
    guard pTest > 0 else { return nil }
    var count = 0
    for y in 0 ..< h - pTest {
        for x in 0 ..< w - pTest where mask[y * w + x] != 0 { count += 1 }
    }
    guard count > 4, var rect = boundingRect(mask, w: w, h: h) else { return nil }

    let px = minWidth - rect.width
    if px > 0 {
        if rect.x1 >= px { rect.x1 -= px } else { rect.x2 += px }
    }
    let py = minSize - rect.height
    if py > 0 {
        if rect.y1 >= py { rect.y1 -= py } else { rect.y2 += py }
    }
    guard rect.x1 >= 0 && rect.y1 >= 0 && rect.x2 <= w && rect.y2 <= h else {
        throw Err("expanded crop outside canvas for \(tag): \(rect.array)")
    }
    let croppedDepth = depthImage.cropped(rect)
    let depth = croppedDepth.px.map { Float($0) / Float(255) }
    return PostPart(tag: tag, image: image.cropped(rect), depth: depth, rect: rect)
}

private func splitEyes(_ parts: inout OrderedParts) {
    guard let part = parts.remove("eyes") else { return }
    let cc = connectedComponents(alphaMask(part.image), w: part.image.w, h: part.image.h)
    guard cc.stats.count > 2 else {
        parts.insert(part)
        return
    }
    guard cc.stats.count >= 5 else { return }
    let order = statsOrderDescending(cc.stats)
    let pairs = [("eyer", "eyel", order[0], order[1]), ("browr", "browl", order[2], order[3])]
    for (name1, name2, id1, id2) in pairs {
        let split = labelLRSplit(cc, id1, id2)
        let first = processCuts(part, stats: split.stats1)
        first.tag = name1
        parts.insert(first)
        let second = processCuts(part, stats: split.stats2)
        second.tag = name2
        parts.insert(second)
    }
}

private func splitTag(_ tag: String, _ parts: inout OrderedParts) {
    guard let part = parts.remove(tag) else { return }
    let cc = connectedComponents(alphaMask(part.image), w: part.image.w, h: part.image.h)
    guard cc.stats.count > 2 else {
        parts.insert(part)
        return
    }
    let order = statsOrderDescending(cc.stats)
    let split = labelLRSplit(cc, order[0], order[1])
    let first = processCuts(part, stats: split.stats1, mask: split.mask1)
    first.tag = tag + "-r"
    parts.insert(first)
    let second = processCuts(part, stats: split.stats2, mask: split.mask2)
    second.tag = tag + "-l"
    parts.insert(second)
}

private func orderedPartTags(infoPath: String) throws -> [String] {
    let text = try String(contentsOfFile: infoPath, encoding: .utf8)
    let pattern = #""parts"\s*:\s*\{"#
    guard let range = text.range(of: pattern, options: .regularExpression) else {
        throw Err("info.json has no parts object: \(infoPath)")
    }
    let bytes = Array(text[range.upperBound...].utf8)
    var keys: [String] = []
    var depth = 1, i = 0
    while i < bytes.count && depth > 0 {
        let c = bytes[i]
        if c == 34 {
            let start = i
            i += 1
            var escaped = false
            while i < bytes.count {
                if escaped { escaped = false }
                else if bytes[i] == 92 { escaped = true }
                else if bytes[i] == 34 { break }
                i += 1
            }
            guard i < bytes.count else { throw Err("unterminated string in \(infoPath)") }
            let literal = Data(bytes[start ... i])
            i += 1
            var j = i
            while j < bytes.count && [9, 10, 13, 32].contains(bytes[j]) { j += 1 }
            if depth == 1 && j < bytes.count && bytes[j] == 58 {
                let wrapped = Data([91]) + literal + Data([93])
                guard let values = try JSONSerialization.jsonObject(with: wrapped) as? [String], let key = values.first else {
                    throw Err("cannot decode part tag in \(infoPath)")
                }
                keys.append(key)
            }
            continue
        }
        if c == 123 || c == 91 { depth += 1 }
        else if c == 125 || c == 93 { depth -= 1 }
        i += 1
    }
    return keys
}

private func replaceRGBFromFullpage(_ part: PostPart, fullpage: RGBAImage) {
    let r = part.rect
    guard r.x1 >= 0 && r.y1 >= 0 && r.x2 <= fullpage.w && r.y2 <= fullpage.h else { return }
    for y in 0 ..< part.image.h {
        for x in 0 ..< part.image.w {
            let dst = (y * part.image.w + x) * 4
            let src = ((r.y1 + y) * fullpage.w + r.x1 + x) * 4
            part.image.px[dst] = fullpage.px[src]
            part.image.px[dst + 1] = fullpage.px[src + 1]
            part.image.px[dst + 2] = fullpage.px[src + 2]
        }
    }
}

struct PostProcessResult {
    var parts: OrderedParts
    var sortedLayers: [PostPart]
    var fullpage: RGBAImage
    var droppedEmptyTags: [String]
}

enum PostProcessor {
    static func run(src: String, tblrSplit: Bool) throws -> PostProcessResult {
        let fullpage = try RGBAImage.load(src + "/src_img.png")
        let infoPath = src + "/info.json"
        let info = try jsonLoad(infoPath)
        guard let infoParts = info["parts"] as? [String: Any] else { throw Err("invalid info.json parts") }
        let tags = try orderedPartTags(infoPath: infoPath)
        guard tags.count == infoParts.count else { throw Err("cannot preserve info.json parts order") }

        var parts = OrderedParts()
        var dropped: [String] = []
        var partID = 0
        for tag in tags {
            if let part = try loadPart(src: src, tag: tag) {
                part.partID = partID
                partID += 1
                parts.insert(part)
            } else {
                dropped.append(tag)
            }
        }

        splitEyes(&parts)
        if tblrSplit {
            guard parts["hair"] == nil else { throw Err("LaMa hair inpainting is outside M5 scope") }
            splitTag("handwear", &parts)
            for tag in ["eyewhite", "irides", "eyelash", "eyebrow"] { splitTag(tag, &parts) }
            splitTag("ears", &parts)
        }
        for tag in ["nose", "mouth"] {
            if let part = parts[tag] { replaceRGBFromFullpage(part, fullpage: fullpage) }
        }
        for part in parts.values {
            let oldImage = part.image
            let oldDepth = part.depth
            let mask = alphaMask(oldImage)
            guard let crop = boundingRect(mask, w: oldImage.w, h: oldImage.h) else { continue }
            var values: [Float] = []
            for i in 0 ..< mask.count where mask[i] != 0 { values.append(oldDepth[i]) }
            part.depthMedian = medianF32(values)
            part.image = oldImage.cropped(crop)
            part.depth = cropValues(oldDepth, w: oldImage.w, h: oldImage.h, r: crop)
            let r = part.rect
            part.rect = PartRect(x1: r.x1 + crop.x1, y1: r.y1 + crop.y1,
                                 x2: r.x1 + crop.x2, y2: r.y1 + crop.y2)
        }

        if let face = parts["face"] {
            for tag in ["nose", "mouth", "eyes"] {
                if let part = parts[tag], part.depthMedian > face.depthMedian {
                    part.depthMedian = face.depthMedian - Float(0.001)
                }
            }
            for tag in ["earr", "earl", "ears"] {
                if let part = parts[tag] { part.depthMedian = face.depthMedian + Float(0.001) }
            }
        }
        let sorted = parts.values.enumerated().sorted { a, b in
            if a.element.depthMedian != b.element.depthMedian {
                return a.element.depthMedian > b.element.depthMedian
            }
            return a.offset < b.offset
        }.map(\.element)
        return PostProcessResult(parts: parts, sortedLayers: sorted, fullpage: fullpage, droppedEmptyTags: dropped)
    }

    static func sidecarJSON(_ result: PostProcessResult) -> [String: Any] {
        var dict: [String: Any] = [:]
        for part in result.parts.values {
            var value: [String: Any] = ["xyxy": part.rect.array, "tag": part.tag]
            if let id = part.partID { value["part_id"] = id }
            value["depth_median"] = Double(part.depthMedian)
            dict[part.tag] = value
        }
        return ["parts": dict, "frame_size": [result.fullpage.h, result.fullpage.w]]
    }
}

enum M5Commands {
    static func post(src: String, psdPath: String, tblrSplit: Bool, out: String?) throws {
        try requireFiles([src + "/info.json", src + "/src_img.png"], hint: "M5 frozen intermediate products")
        let resolvedPSDPath = psdPath.hasPrefix("/") ? psdPath : FileManager.default.currentDirectoryPath + "/" + psdPath
        let start = nowMs()
        let result = try PostProcessor.run(src: src, tblrSplit: tblrSplit)
        let processMs = nowMs() - start
        try PSDWriter.write(layers: result.sortedLayers, canvasW: result.fullpage.w,
                            canvasH: result.fullpage.h, path: resolvedPSDPath)
        let sidecar = PostProcessor.sidecarJSON(result)
        jsonWriteM5(sidecar, resolvedPSDPath + ".json")
        let elapsed = nowMs() - start
        let rows = result.sortedLayers.map {
            ["tag": $0.tag, "xyxy": $0.rect.array, "depth_median": Double($0.depthMedian)] as [String: Any]
        }
        jsonWriteM5([
            "command": "m5-post", "source": src, "psd": resolvedPSDPath,
            "rotate": false, "save_to_psd": true, "tblr_split": tblrSplit,
            "frame_size": [result.fullpage.h, result.fullpage.w],
            "input_part_slots": (try? orderedPartTags(infoPath: src + "/info.json"))?.count ?? -1,
            "dropped_empty_tags": result.droppedEmptyTags,
            "layer_count": rows.count, "layer_order": rows.map { $0["tag"] as! String },
            "layers": rows, "process_ms": processMs, "elapsed_ms": elapsed,
        ], out)
        print(String(format: "m5-post: %d layers, canvas %dx%d, %.1f ms -> %@",
                     rows.count, result.fullpage.w, result.fullpage.h, elapsed, resolvedPSDPath))
        print("layer order: " + rows.map { $0["tag"] as! String }.joined(separator: " | "))
    }
}
