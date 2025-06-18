import SwiftUI

struct ContentView: View {
    @State private var inputText: String = ""
    @State private var predictions: [(String, Float)] = []
    @State private var errorMessage: String?

    var body: some View {
        VStack(spacing: 20) {
            Text("ONNX Text Classifier")
                .font(.title)
                .padding(.top)

            TextField("Enter your text here", text: $inputText)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .padding(.horizontal)

            Button("Classify") {
                classifyText()
            }
            .padding()
            .background(Color.blue)
            .foregroundColor(.white)
            .cornerRadius(10)

            if let error = errorMessage {
                Text("❌ \(error)")
                    .foregroundColor(.red)
            }

            List(predictions, id: \.0) { item in
                HStack {
                    Text(item.0)
                        .fontWeight(.bold)
                    Spacer()
                    Text(String(format: "%.2f", item.1))
                }
            }
        }
        .padding()
    }

    private func classifyText() {
        do {
            let runner = try ONNXModelRunner()
            let result = try runner.predict(text: inputText)
            self.predictions = result.sorted { $0.1 > $1.1 }
            self.errorMessage = nil
        } catch {
            self.errorMessage = error.localizedDescription
            self.predictions = []
        }
    }
}

#Preview {
    ContentView()
}
