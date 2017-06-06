import Foundation

// Perceptron


func randomDouble() -> Double {
    return Double(arc4random()) / Double(UINT32_MAX)
}

class Perceptron {
    var bias: Double
    private var offState: Int = 0
    let learningRate = 0.01
    
    // weights[0] is the weight for the bias input
    var weights: [Double]
    
    
    init(numInputs: Int, offState: Int, bias: Double) {
        self.offState = offState
        self.bias = bias
        
        self.weights = [Double]()
        for _ in 0...numInputs {
            weights.append(1.0 * (randomDouble() - 0.5))
        }
    }
    
    init(numInputs: Int, bias: Double) {
        self.bias = bias
        
        self.weights = [Double]()
        for _ in 0...numInputs{
            weights.append(1.0 * (randomDouble() - 0.5))
        }
    }
    
    func feedForward(input: [Double]) -> Int {
        assert(input.count + 1 == weights.count)
        
        var sum = 0.0
        for i in 0..<input.count {
            sum += input[i] * weights[i + 1]
        }
        sum += bias * weights[0]
        return (sum > 0) ? 1 : offState
    }
    
    func backProp(input: [Double], output: Int) -> Int {
        let prediction = feedForward(input: input)
        let error = output - prediction
        
        for i in 0..<input.count {
            weights[i + 1] += learningRate * Double(error) * input[i]
        }
        weights[0] += learningRate * Double(error) * bias
        
        return error
    }
}


// can we predict if a given x value is above or below the line y=3x+1 ?
var line = { (x: Double) in return 3 * x + 1 }

var isAbove = { (x: Double) in return line(x) > 0 ? 1 : -1 }

var p = Perceptron(numInputs: 1, offState: -1, bias: 1)

struct PerceptronDataPair {
    let input: [Double]
    let output: Int
}

struct PerceptronTrainer {
    let data: [PerceptronDataPair]
    
    func train( p: inout Perceptron) -> Int {
        var error: Int = 0
        var count = 0
        
        for d in data {
            error = p.backProp(input: d.input, output: d.output)
            
            if (count % 5 == 0) {
                print("Iter: \(count), Error: \(error)")
            }
            count += 1
        }
        
        return error
    }
}

// create training data
func createData(numPoints: Int) -> [PerceptronDataPair] {
    var data = [PerceptronDataPair]()
    
    for _ in 0..<numPoints {
        let x = [2.0 * (randomDouble() - 0.5)]
        let y = line(x[0])
        data.append(PerceptronDataPair(input: x, output: isAbove(y)))
    }

    return data
}

let trainingData = createData(numPoints: 100)


let trainer = PerceptronTrainer(data: trainingData)
trainer.train(p: &p)

// let's see how good it works

let testData = createData(numPoints: 100)

func evaluatePerceptron(p: Perceptron, testData: [PerceptronDataPair]) -> Double {
    var correct = 0
    for d in testData {
        let prediction = p.feedForward(input: d.input)
        if (prediction == d.output) {
            correct += 1
        }
    }
    
    return Double(correct) / Double(testData.count)
}

// The % correct will be much higher than for an untrained perceptron
evaluatePerceptron(p: p, testData: testData)

let pUntrained = Perceptron(numInputs: 1, offState: -1, bias: 1)
evaluatePerceptron(p: pUntrained, testData: testData)



