import Foundation
import onnxruntime_objc

class Tokenizer {
    private var vocab: [String: Int]

    init(vocab: [String: Int]) {
        self.vocab = vocab
    }

    func tokenize(text: String) -> [Int] {
        let words = text.lowercased().split(separator: " ").map { String($0) }
        return words.map { vocab[$0] ?? vocab["<OOV>"] ?? 1 }
    }
}

class LabelVocabLoader {
    private(set) var labelMap: [Int: String] = [:]
    private(set) var vocab: [String: Int] = [:]

    init(labelMapPath: String, vocabPath: String) throws {
        let labelMapData = try Data(contentsOf: URL(fileURLWithPath: labelMapPath))
        let vocabData = try Data(contentsOf: URL(fileURLWithPath: vocabPath))

        if let labelMapJson = try JSONSerialization.jsonObject(with: labelMapData) as? [String: String] {
            for (key, value) in labelMapJson {
                if let intKey = Int(key) {
                    labelMap[intKey] = value
                }
            }
        }

        if let vocabJson = try JSONSerialization.jsonObject(with: vocabData) as? [String: Int] {
            vocab = vocabJson
        }
    }
}

class ONNXModelRunner {
    private var session: ORTSession
    private var labelMap: [Int: String]
    private var vocab: [String: Int]
    private let maxLen = 30

    init() throws {
        guard let modelPath = Bundle.main.path(forResource: "model", ofType: "onnx"),
              let labelPath = Bundle.main.path(forResource: "label_map", ofType: "json"),
              let vocabPath = Bundle.main.path(forResource: "vocab", ofType: "json") else {
            throw NSError(domain: "Paths", code: 1, userInfo: [NSLocalizedDescriptionKey: "Missing model or json files"])
        }

        let loader = try LabelVocabLoader(labelMapPath: labelPath, vocabPath: vocabPath)
        self.labelMap = loader.labelMap
        self.vocab = loader.vocab

        let env = try ORTEnv(loggingLevel: .warning)
        let sessionOptions = try ORTSessionOptions()
        self.session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: sessionOptions)

        print("✅ Model loaded.")
        print("Input names: \(try session.inputNames())")
        print("Output names: \(try session.outputNames())")
    }

    func predict(text: String) throws -> [(String, Float)] {
        let tokenizer = Tokenizer(vocab: vocab)
        var tokens = tokenizer.tokenize(text: text)

        if tokens.count < maxLen {
            tokens += Array(repeating: 0, count: maxLen - tokens.count)
        } else if tokens.count > maxLen {
            tokens = Array(tokens.prefix(maxLen))
        }

        let int32Tokens = tokens.map { Int32($0) }
        let inputData = Data(from: int32Tokens)

        let inputTensor = try ORTValue(
            tensorData: NSMutableData(data: inputData),
            elementType: .int32,
            shape: [1, NSNumber(value: maxLen)]
        )

        let inputs: [String: ORTValue] = [
            "input": inputTensor
        ]

        let outputs = try session.run(
            withInputs: inputs,
            outputNames: Set(["sequential"]),
            runOptions: nil
        )

        guard let ortValue = outputs["sequential"] else {
            throw NSError(domain: "ONNX", code: 2, userInfo: [NSLocalizedDescriptionKey: "Missing 'sequential' output"])
        }

        let rawData = try ortValue.tensorData()

        let resultData: Data
        if let d = rawData as? Data {
            resultData = d
        } else {
            throw NSError(domain: "ONNX", code: 4, userInfo: [NSLocalizedDescriptionKey: "Output is not valid Data"])
        }

        let classCount = labelMap.count
        let scores: [Float] = resultData.withUnsafeBytes { buffer in
            let pointer = buffer.bindMemory(to: Float.self)
            return Array(pointer.prefix(classCount))
        }

        return scores.enumerated().map { (i, score) in
            (labelMap[i] ?? "Unknown", score)
        }
    }
}

// 🔁 Розширення Data
private extension Data {
    init<T>(from array: [T]) {
        self = array.withUnsafeBufferPointer { Data(buffer: $0) }
    }
}
