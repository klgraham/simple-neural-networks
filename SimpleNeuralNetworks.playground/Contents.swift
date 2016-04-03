//: Playground - noun: a place where people can play

import UIKit

// Perceptron

let bias = 1.0
var biasWeight = 1.0
var x: [Double] = [12, 4]
var w: [Double] = [0.5, -1]

func step(z: Double) -> Int {
    return (z >= 0) ? 1 : 0
}

func activate(bias: Double, bW: Double, x: [Double], w: [Double]) -> Int {
    var sum = 0.0
    for i in 0..<x.count {
        sum += x[i] * w[i]
    }
    sum += bias * bW
    return step(sum)
}

activate(bias, bW: biasWeight, x: x, w: w)

struct Perceptron {
    var bias: Double
    private var offState: Int = 0
    let learningRate = 0.01
    
    // weights[0] is the weight for the bias input
    var weights: [Double]
    
    func feedForward(input: [Double]) -> Int {
        assert(input.count + 1 == weights.count)
        
        var sum = 0.0
        for i in 0..<input.count {
            sum += input[i] * weights[i + 1]
        }
        sum += bias * weights[0]
        return (sum > 0) ? 1 : offState
    }
    
    mutating func backProp(input: [Double], output: Int) {
        let prediction = feedForward(input)
        let error = output - prediction
        
        for i in 0..<weights.count {
            weights[i] += learningRate * Double(error) * input[i]
        }
    }
    
    init(numInputs: Int, offState: Int, bias: Double) {
        self.offState = offState
        self.bias = bias
        
        self.weights = [Double]()
        for _ in 0...numInputs {
            weights.append(1.0 * (drand48() - 0.5))
        }
    }
    
    init(numInputs: Int, bias: Double) {
        self.bias = bias
        
        self.weights = [Double]()
        for _ in 0...numInputs{
            weights.append(1.0 * (drand48() - 0.5))
        }
    }
}

var p = Perceptron(numInputs: 2, offState: -1, bias: 1)
p.feedForward(x)
