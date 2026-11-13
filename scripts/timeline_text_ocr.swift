// Vision OCR for timeline_text.py. stdin JSON lines {"path", "id"?, "crop"?: [x0,y0,x1,y1] fractions, "scale"?, "fix"?}
// stdout JSON lines {"path", "id", "size", "lines": [[text, conf, x, y, w, h, [[word, x, y, w, h], ...]]]}
// (fractions of the crop, origin top left). Build: swiftc -O timeline_text_ocr.swift -o <bin>
import Foundation
import Vision
import ImageIO
import CoreGraphics

func load(_ path: String) -> CGImage? {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil) else { return nil }
    return CGImageSourceCreateImageAtIndex(src, 0, nil)
}

func scaled(_ img: CGImage, _ s: Double) -> CGImage? {
    let w = Int(Double(img.width) * s), h = Int(Double(img.height) * s)
    guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                              space: CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
    ctx.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
    ctx.interpolationQuality = .high
    ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
    return ctx.makeImage()
}

func box(_ b: CGRect) -> [Double] {
    return [Double(b.minX), Double(1 - b.maxY), Double(b.width), Double(b.height)]
}

while let line = readLine() {
    guard let data = line.data(using: .utf8),
          let job = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          let path = job["path"] as? String else { continue }
    var out: [String: Any] = ["path": path]
    if let id = job["id"] { out["id"] = id }
    autoreleasepool {
        guard var img = load(path) else { out["error"] = "load"; return }
        if let c = job["crop"] as? [Double], c.count == 4 {
            let W = Double(img.width), H = Double(img.height)
            let r = CGRect(x: c[0] * W, y: c[1] * H, width: (c[2] - c[0]) * W, height: (c[3] - c[1]) * H).integral
            if let cut = img.cropping(to: r) { img = cut }
        }
        if let s = job["scale"] as? Double, s != 1.0, let big = scaled(img, s) { img = big }
        let req = VNRecognizeTextRequest()
        req.recognitionLevel = .accurate
        req.usesLanguageCorrection = (job["fix"] as? Bool) ?? true
        req.recognitionLanguages = ["en-US"]
        let handler = VNImageRequestHandler(cgImage: img, options: [:])
        do { try handler.perform([req]) } catch { out["error"] = "\(error)"; return }
        var lines: [[Any]] = []
        for o in req.results ?? [] {
            guard let t = o.topCandidates(1).first else { continue }
            let s = t.string
            var words: [[Any]] = []
            var i = s.startIndex
            while i < s.endIndex {
                while i < s.endIndex && s[i] == " " { i = s.index(after: i) }
                var j = i
                while j < s.endIndex && s[j] != " " { j = s.index(after: j) }
                if i < j, let r = try? t.boundingBox(for: i..<j) {
                    words.append([String(s[i..<j])] + box(r.boundingBox))
                }
                i = j
            }
            lines.append([s, Double(t.confidence)] + box(o.boundingBox) + [words])
        }
        out["lines"] = lines
        out["size"] = [img.width, img.height]
    }
    if let d = try? JSONSerialization.data(withJSONObject: out), let s = String(data: d, encoding: .utf8) {
        print(s); fflush(stdout)
    }
}
