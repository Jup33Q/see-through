import CoreGraphics
import Foundation
import ImageIO

/// PNG RGBA8 IO 与 cv2 语义对齐的 resize / head 合成。
/// 注意：golden PNG 由 PIL (numpy uint8, 直通 alpha) 写出；CGImage 的
/// premultiplied 语义需显式处理，避免透明区颜色被预乘破坏。

struct RGBAImage {
    var w: Int
    var h: Int
    var px: [UInt8]  // RGBA 直通 alpha，行优先

    static func load(_ path: String) throws -> RGBAImage {
        let url = URL(fileURLWithPath: path) as CFURL
        guard let src = CGImageSourceCreateWithURL(url, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { throw Err("cannot read png: \(path)") }
        let w = img.width, h = img.height
        guard img.bitsPerComponent == 8, img.bitsPerPixel == 32, let dp = img.dataProvider,
              let data = dp.data else {
            // 兜底：画进 premultiplied 上下文再反解（可能损失透明区颜色精度）
            return try loadViaContext(img, w, h, path)
        }
        let bytes = [UInt8](data as Data)
        let bpr = img.bytesPerRow
        var out = [UInt8](repeating: 0, count: w * h * 4)
        let alphaInfo = img.alphaInfo
        for y in 0 ..< h {
            for x in 0 ..< w {
                let si = y * bpr + x * 4, di = (y * w + x) * 4
                let r = bytes[si], g = bytes[si + 1], b = bytes[si + 2], a = bytes[si + 3]
                switch alphaInfo {
                case .last:  // RGBA 直通
                    out[di] = r; out[di + 1] = g; out[di + 2] = b; out[di + 3] = a
                case .premultipliedLast:
                    if a == 255 { out[di] = r; out[di + 1] = g; out[di + 2] = b }
                    else if a == 0 { out[di] = 0; out[di + 1] = 0; out[di + 2] = 0 }
                    else {  // 反预乘
                        out[di] = UInt8(min(255, (Int(r) * 255 + Int(a) / 2) / Int(a)))
                        out[di + 1] = UInt8(min(255, (Int(g) * 255 + Int(a) / 2) / Int(a)))
                        out[di + 2] = UInt8(min(255, (Int(b) * 255 + Int(a) / 2) / Int(a)))
                    }
                    out[di + 3] = a
                default:
                    throw Err("unsupported alphaInfo \(alphaInfo.rawValue) in \(path)")
                }
            }
        }
        return RGBAImage(w: w, h: h, px: out)
    }

    private static func loadViaContext(_ img: CGImage, _ w: Int, _ h: Int, _ path: String) throws -> RGBAImage {
        var buf = [UInt8](repeating: 0, count: w * h * 4)
        guard let ctx = CGContext(data: &buf, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw Err("cannot create ctx for \(path)") }
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
        // premultiplied → 直通
        for i in stride(from: 0, to: buf.count, by: 4) {
            let a = Int(buf[i + 3])
            if a == 0 { buf[i] = 0; buf[i + 1] = 0; buf[i + 2] = 0 }
            else if a != 255 {
                for c in 0 ..< 3 { buf[i + c] = UInt8(min(255, (Int(buf[i + c]) * 255 + a / 2) / a)) }
            }
        }
        return RGBAImage(w: w, h: h, px: buf)
    }

    func save(_ path: String) throws {
        let data = Data(px)
        guard let provider = CGDataProvider(data: data as CFData) else { throw Err("provider") }
        guard let img = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32,
                                bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                                provider: provider, decode: nil, shouldInterpolate: false,
                                intent: .defaultIntent)
        else { throw Err("cgimage create failed") }
        try FileManager.default.createDirectory(
            atPath: URL(fileURLWithPath: path).deletingLastPathComponent().path, withIntermediateDirectories: true)
        guard let dst = CGImageDestinationCreateWithURL(
            URL(fileURLWithPath: path) as CFURL, "public.png" as CFString, 1, nil)
        else { throw Err("dest create failed: \(path)") }
        CGImageDestinationAddImage(dst, img, nil)
        guard CGImageDestinationFinalize(dst) else { throw Err("png write failed: \(path)") }
    }
}

// MARK: - resize（cv2 语义：降采样 INTER_AREA=面积平均，升采样 INTER_LINEAR=双线性）

func smartResize(_ img: RGBAImage, _ tw: Int, _ th: Int) -> RGBAImage {
    if tw == img.w && th == img.h { return img }
    if th * tw < img.h * img.w { return areaResize(img, tw, th) }
    return bilinearResize(img, tw, th)
}

/// 可分离面积平均（cv2 INTER_AREA 等价，浮点实现）
private func areaResize(_ img: RGBAImage, _ tw: Int, _ th: Int) -> RGBAImage {
    let (w, h) = (img.w, img.h)
    let sx = Double(w) / Double(tw), sy = Double(h) / Double(th)
    // 水平: (h, tw, 4) float
    var tmp = [Float](repeating: 0, count: h * tw * 4)
    for y in 0 ..< h {
        for dx in 0 ..< tw {
            let x0 = Double(dx) * sx, x1 = Double(dx + 1) * sx
            var acc = (Float(0), Float(0), Float(0), Float(0))
            var ix = Int(x0)
            while ix < w && Double(ix) < x1 {
                let cover = min(x1, Double(ix + 1)) - max(x0, Double(ix))
                if cover > 0 {
                    let si = (y * w + ix) * 4
                    acc.0 += Float(cover) * Float(img.px[si])
                    acc.1 += Float(cover) * Float(img.px[si + 1])
                    acc.2 += Float(cover) * Float(img.px[si + 2])
                    acc.3 += Float(cover) * Float(img.px[si + 3])
                }
                ix += 1
            }
            let di = (y * tw + dx) * 4
            let inv = Float(1.0 / sx)
            tmp[di] = acc.0 * inv; tmp[di + 1] = acc.1 * inv
            tmp[di + 2] = acc.2 * inv; tmp[di + 3] = acc.3 * inv
        }
    }
    // 垂直: (th, tw, 4) → uint8
    var out = [UInt8](repeating: 0, count: th * tw * 4)
    for dy in 0 ..< th {
        let y0 = Double(dy) * sy, y1 = Double(dy + 1) * sy
        for x in 0 ..< tw {
            var acc = (Float(0), Float(0), Float(0), Float(0))
            var iy = Int(y0)
            while iy < h && Double(iy) < y1 {
                let cover = min(y1, Double(iy + 1)) - max(y0, Double(iy))
                if cover > 0 {
                    let si = (iy * tw + x) * 4
                    acc.0 += Float(cover) * tmp[si]
                    acc.1 += Float(cover) * tmp[si + 1]
                    acc.2 += Float(cover) * tmp[si + 2]
                    acc.3 += Float(cover) * tmp[si + 3]
                }
                iy += 1
            }
            let di = (dy * tw + x) * 4
            let inv = Float(1.0 / sy)
            out[di] = u8round(acc.0 * inv); out[di + 1] = u8round(acc.1 * inv)
            out[di + 2] = u8round(acc.2 * inv); out[di + 3] = u8round(acc.3 * inv)
        }
    }
    return RGBAImage(w: tw, h: th, px: out)
}

/// 双线性（cv2 INTER_LINEAR 等价，中心对齐映射）
private func bilinearResize(_ img: RGBAImage, _ tw: Int, _ th: Int) -> RGBAImage {
    let (w, h) = (img.w, img.h)
    let sx = Double(w) / Double(tw), sy = Double(h) / Double(th)
    var out = [UInt8](repeating: 0, count: th * tw * 4)
    for dy in 0 ..< th {
        let fy = max(0.0, (Double(dy) + 0.5) * sy - 0.5)
        let y0 = min(Int(fy), h - 1), y1 = min(y0 + 1, h - 1)
        let wy = Float(fy - Double(y0))
        for dx in 0 ..< tw {
            let fx = max(0.0, (Double(dx) + 0.5) * sx - 0.5)
            let x0 = min(Int(fx), w - 1), x1 = min(x0 + 1, w - 1)
            let wx = Float(fx - Double(x0))
            let i00 = (y0 * w + x0) * 4, i01 = (y0 * w + x1) * 4
            let i10 = (y1 * w + x0) * 4, i11 = (y1 * w + x1) * 4
            let di = (dy * tw + dx) * 4
            for c in 0 ..< 4 {
                let top = Float(img.px[i00 + c]) * (1 - wx) + Float(img.px[i01 + c]) * wx
                let bot = Float(img.px[i10 + c]) * (1 - wx) + Float(img.px[i11 + c]) * wx
                out[di + c] = u8round(top * (1 - wy) + bot * wy)
            }
        }
    }
    return RGBAImage(w: tw, h: th, px: out)
}

private func u8round(_ v: Float) -> UInt8 {
    UInt8(min(255, max(0, Int((v + 0.5).rounded(.down)))))
}

/// head pass 合成（inference_utils.py:172-183 + dump 脚本）：
/// resize 1280² → (th=scale_size[0], tw=scale_size[1])，裁 [py1:py2, px1:px2]，
/// 贴到 1280² 透明画布的 paste_xy=[px1f, py1f]（x=列, y=行）。
/// 注: scale_size_wh 源于 center_square_pad_resize 的方形 pad_size，w==h，顺序无歧义。
func compositeHeadFrame(_ frame: RGBAImage, composite c: [String: Any]) -> RGBAImage {
    let ss = (c["scale_size_wh"] as! [Any]).map { ($0 as! NSNumber).intValue }
    let cr = c["crop_after_resize"] as! [String: Any]
    let py1 = (cr["py1"] as! NSNumber).intValue, py2 = (cr["py2"] as! NSNumber).intValue
    let px1 = (cr["px1"] as! NSNumber).intValue, px2 = (cr["px2"] as! NSNumber).intValue
    let paste = (c["paste_xy"] as! [Any]).map { ($0 as! NSNumber).intValue }
    let resized = smartResize(frame, ss[1], ss[0])  // (th=ss[0], tw=ss[1])，方形无差
    let cw = px2 - px1, ch = py2 - py1
    var canvas = RGBAImage(w: frame.w, h: frame.h, px: [UInt8](repeating: 0, count: frame.w * frame.h * 4))
    for r in 0 ..< ch {
        let srcRow = ((py1 + r) * resized.w + px1) * 4
        let dstRow = ((paste[1] + r) * canvas.w + paste[0]) * 4
        canvas.px.replaceSubrange(dstRow ..< dstRow + cw * 4,
                                  with: resized.px[srcRow ..< srcRow + cw * 4])
    }
    return canvas
}
