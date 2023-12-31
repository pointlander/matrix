// Copyright 2023 The Matrix Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package matrix

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	"github.com/pointlander/matrix/vector"
)

const (
	// S is the scaling factor for the softmax
	S = 1.0 - 1e-300
	// Window is the window size
	Window = 64
	// Samples is the number of samples to take
	Samples = 1024
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

// Random is a random variable
type Random struct {
	Mean   float32
	StdDev float32
}

// RandomMatrix is a random matrix
type RandomMatrix struct {
	Cols int
	Rows int
	Data []Random
}

// NewRandomMatrix returns a new random matrix
func NewRandomMatrix(cols, rows int) RandomMatrix {
	data := make([]Random, cols*rows)
	factor := float32(math.Sqrt(2.0 / float64(cols)))
	for i := range data {
		data[i].StdDev = factor
	}
	return RandomMatrix{
		Cols: cols,
		Rows: rows,
		Data: data,
	}
}

// Sample samples a matrix
func (r RandomMatrix) Sample(rng *rand.Rand) Matrix {
	sample := NewMatrix(0, r.Cols, r.Rows)
	for _, v := range r.Data {
		sample.Data = append(sample.Data, float32(rng.NormFloat64())*v.StdDev+v.Mean)
	}
	return sample
}

// Matrix is a float32 matrix
type Matrix struct {
	Cols   int
	Rows   int
	Data   []float32
	States [][]float32
}

// NewMatrix32 creates a new float32 matrix
func NewMatrix(states, cols, rows int) Matrix {
	m := Matrix{
		Cols: cols,
		Rows: rows,
		Data: make([]float32, 0, cols*rows),
	}
	if states > 0 {
		m.States = make([][]float32, states)
		for i := range m.States {
			m.States[i] = make([]float32, cols*rows)
		}
	}
	return m
}

// Size is the size of the float32 matrix
func (m Matrix) Size() int {
	return m.Cols * m.Rows
}

// String returns a string for the matrix
func (m Matrix) String() string {
	s := "\n"
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			s += fmt.Sprintf(" %f", m.Data[i*m.Cols+j])
		}
		s += "\n"
	}
	return s + "\n"
}

// MulT multiplies two matrices and computes the transpose
func MulT(m Matrix, n Matrix) Matrix {
	if m.Cols != n.Cols {
		panic(fmt.Errorf("%d != %d", m.Cols, n.Cols))
	}
	columns := m.Cols
	o := Matrix{
		Cols: m.Rows,
		Rows: n.Rows,
		Data: make([]float32, 0, m.Rows*n.Rows),
	}
	lenn, lenm := len(n.Data), len(m.Data)
	for i := 0; i < lenn; i += columns {
		nn := n.Data[i : i+columns]
		for j := 0; j < lenm; j += columns {
			mm := m.Data[j : j+columns]
			o.Data = append(o.Data, vector.Dot(mm, nn))
		}
	}
	return o
}

// Mul is regular matrix multiplication
func Mul(m Matrix, n Matrix) Matrix {
	return MulT(T(n), m)
}

// Add adds two float32 matrices
func Add(m Matrix, n Matrix) Matrix {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float32, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value+n.Data[i%lenb])
	}
	return o
}

// Sub subtracts two float32 matrices
func Sub(m Matrix, n Matrix) Matrix {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float32, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value-n.Data[i%lenb])
	}
	return o
}

// Sigmoid computes the sigmoid of a matrix
func Sigmoid(m Matrix) Matrix {
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float32, 0, m.Cols*m.Rows),
	}
	for _, value := range m.Data {
		o.Data = append(o.Data, float32(1/(1+math.Exp(-float64(value)))))
	}
	return o
}

// Step computes the step function of a float32 matrix
func Step(m Matrix) Matrix {
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float32, 0, m.Cols*m.Rows),
	}
	for _, value := range m.Data {
		if value > 0 {
			value = 1
		} else {
			value = -1
		}
		o.Data = append(o.Data, value)
	}
	return o
}

// Quadratic computes the quadratic loss of two matrices
func Quadratic(m Matrix, n Matrix) Matrix {
	size, width := len(m.Data), m.Cols
	o := Matrix{
		Cols: m.Rows,
		Rows: 1,
		Data: make([]float32, 0, 1*m.Rows),
	}
	for i := 0; i < size; i += width {
		a, b, sum := m.Data[i:i+width], n.Data[i:i+width], float32(0.0)
		for j, ax := range a {
			diff := ax - b[j]
			sum += diff * diff
		}
		o.Data = append(o.Data, float32(math.Sqrt(float64(sum))))
	}
	return o
}

// QuadraticSet computes the quadratic loss of sub sets of two matrices
func QuadraticSet(m Matrix, n Matrix, set []int) Matrix {
	o := Matrix{
		Cols: 1,
		Rows: 1,
		Data: make([]float32, 0, 1),
	}
	sum := float32(0.0)
	for _, i := range set {
		diff := m.Data[i] - n.Data[i]
		sum += diff * diff
	}
	o.Data = append(o.Data, float32(math.Sqrt(float64(sum))))
	return o
}

// Avg computes the avg of a matrix
func Avg(m Matrix) Matrix {
	o := Matrix{
		Cols: 1,
		Rows: 1,
		Data: make([]float32, 0, 1),
	}
	sum := float32(0.0)
	for _, value := range m.Data {
		sum += value
	}
	o.Data = append(o.Data, sum/float32(len(m.Data)))
	return o
}

// T tramsposes a matrix
func T(m Matrix) Matrix {
	o := Matrix{
		Cols: m.Rows,
		Rows: m.Cols,
		Data: make([]float32, 0, m.Cols*m.Rows),
	}
	for i := 0; i < m.Cols; i++ {
		for j := 0; j < m.Rows; j++ {
			o.Data = append(o.Data, m.Data[j*m.Cols+i])
		}
	}
	return o
}

// Normalize normalizes a matrix to the unit vector
func Normalize(m Matrix) Matrix {
	size, width := len(m.Data), m.Cols
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float32, 0, m.Cols*m.Rows),
	}
	for i := 0; i < size; i += width {
		sum := float32(0.0)
		for _, ax := range m.Data[i : i+width] {
			sum += ax * ax
		}
		length := float32(math.Sqrt(float64(sum)))
		if sum == 0 {
			length = 1
		}
		for _, ax := range m.Data[i : i+width] {
			o.Data = append(o.Data, ax/length)
		}
	}
	return o
}

func softmax(values []float32) {
	max := float32(0.0)
	for _, v := range values {
		if v > max {
			max = v
		}
	}
	s := max * S
	sum := float32(0.0)
	for j, value := range values {
		values[j] = float32(math.Exp(float64(value - s)))
		sum += values[j]
	}
	for j, value := range values {
		values[j] = value / sum
	}
}

// SelfAttention computes the self attention of Q, K, V
func SelfAttention(Q, K, V Matrix) Matrix {
	o := Matrix{
		Cols: V.Rows,
		Rows: K.Rows,
		Data: make([]float32, 0, V.Rows*K.Rows),
	}
	outputs, values := make([]float32, V.Cols), make([]float32, Q.Rows)
	V = T(V)
	for i := 0; i < K.Rows; i++ {
		K := K.Data[i*K.Cols : (i+1)*K.Cols]
		for j := 0; j < Q.Rows; j++ {
			Q := Q.Data[j*Q.Cols : (j+1)*Q.Cols]
			values[j] = vector.Dot(K, Q)
		}
		softmax(values)

		for j := 0; j < V.Rows; j++ {
			V := V.Data[j*V.Cols : (j+1)*V.Cols]
			outputs[j] = vector.Dot(values, V)
		}
		softmax(outputs)
		o.Data = append(o.Data, outputs...)
	}
	return o
}

// SelfEntropy computes the self entropy of Q, K, V
func SelfEntropy(Q, K, V Matrix) ([]Matrix, []float32) {
	entropies, values, results := make([]float32, V.Cols), make([]float32, K.Rows), make([]float32, 0, K.Rows)
	outputs := make([]Matrix, 0, 8)
	V = T(V)
	for i := 0; i < K.Rows; i++ {
		K := K.Data[i*K.Cols : (i+1)*K.Cols]
		for j := 0; j < Q.Rows; j++ {
			Q := Q.Data[j*Q.Cols : (j+1)*Q.Cols]
			values[j] = vector.Dot(K, Q)
		}
		softmax(values)

		for j := 0; j < V.Rows; j++ {
			V := V.Data[j*V.Cols : (j+1)*V.Cols]
			entropies[j] = vector.Dot(values, V)
		}
		output := NewMatrix(0, V.Cols, 1)
		output.Data = append(output.Data, entropies...)
		outputs = append(outputs, output)
		softmax(entropies)

		entropy := 0.0
		for _, e := range entropies {
			entropy += float64(e) * math.Log(float64(e))
		}
		results = append(results, float32(-entropy))
	}
	return outputs, results
}

// Everett is the everett activation function
func Everett(m Matrix) Matrix {
	o := Matrix{
		Cols: 2 * m.Cols,
		Rows: m.Rows,
		Data: make([]float32, 0, 2*m.Cols*m.Rows),
	}
	for _, value := range m.Data {
		min, max := value, value
		if min > 0 {
			min = 0
		}
		if max < 0 {
			max = 0
		}
		o.Data = append(o.Data, min, max)
	}
	return o
}

// TaylorSoftmax is the taylor softmax
// https://arxiv.org/abs/1511.05042
func TaylorSoftmax(m Matrix) Matrix {
	o := Matrix{
		Cols: m.Cols,
		Rows: m.Rows,
		Data: make([]float32, 0, m.Cols*m.Rows),
	}
	var sum float32
	columns, lenm := m.Cols, len(m.Data)
	for i := 0; i < lenm; i += columns {
		nn := m.Data[i : i+columns]
		for _, v := range nn {
			sum += 1 + v + v*v/2
		}
	}
	for i := 0; i < lenm; i += columns {
		nn := m.Data[i : i+columns]
		for _, v := range nn {
			o.Data = append(o.Data, (1+v+v*v/2)/sum)
		}
	}
	return o
}

// LU lower upper decomposition
// https://www.geeksforgeeks.org/doolittle-algorithm-lu-decomposition/
func LU(mat Matrix) (Matrix, Matrix) {
	n := mat.Cols
	lower := NewMatrix(0, n, n)
	lower.Data = lower.Data[:n*n]
	upper := NewMatrix(0, n, n)
	upper.Data = upper.Data[:n*n]
	for i := 0; i < n; i++ {
		for k := i; k < n; k++ {
			sum := float32(0.0)
			for j := 0; j < i; j++ {
				sum += lower.Data[i*n+j] * upper.Data[j*n+k]
			}
			upper.Data[i*n+k] = mat.Data[i*n+k] - sum
		}
		for k := i; k < n; k++ {
			if i == k {
				lower.Data[i*n+i] = 1
			} else {
				sum := float32(0.0)
				for j := 0; j < i; j++ {
					sum += lower.Data[k*n+j] * upper.Data[j*n+i]
				}
				if upper.Data[i*n+i] != 0 {
					lower.Data[k*n+i] = (mat.Data[k*n+i] - sum) / upper.Data[i*n+i]
				}
			}
		}
	}
	return lower, upper
}

// Determinant calculates the determinant of a matrix
func Determinant(a Matrix) (float32, error) {
	l, u := LU(a)
	det := float32(1)
	for i := 0; i < l.Cols; i++ {
		det *= l.Data[i*l.Cols+i] * u.Data[i*l.Cols+i]
	}
	return det, nil
}

// Multi is a multivariate distribution
type Multi struct {
	E Matrix
	U Matrix
	A Matrix
}

// NewMulti make a new multi
func NewMulti(vars int) Multi {
	factor := float32(math.Sqrt(2.0 / float64(vars)))
	a := NewMatrix(0, vars, vars)
	a.Data = a.Data[:cap(a.Data)]
	for i := 0; i < vars; i++ {
		for j := 0; j < vars; j++ {
			if i == j {
				a.Data[i*vars+j] = factor
			}
		}
	}
	u := NewMatrix(1, vars, 1)
	u.Data = u.Data[:cap(u.Data)]
	return Multi{
		A: a,
		U: u,
	}
}

// NewMultiFromData creates a new multivariate distribution
func NewMultiFromData(vars [][]float32) Multi {
	length := len(vars)
	e := NewMatrix(1, length, length)
	e.Data = e.Data[:cap(e.Data)]
	u := NewMatrix(1, length, 1)
	u.Data = u.Data[:cap(u.Data)]
	for i, v := range vars {
		for _, vv := range v {
			u.Data[i] += vv
		}
	}
	size := len(vars[0])
	for i := range u.Data {
		u.Data[i] /= float32(size)
	}
	for i := 0; i < length; i++ {
		for j := i; j < length; j++ {
			for k := 0; k < size; k++ {
				e.Data[i*length+j] += (vars[i][k] - u.Data[i]) * (vars[j][k] - u.Data[j])
			}
			e.Data[i*length+j] /= float32(size)
		}
	}
	for i := 0; i < length; i++ {
		for j := i + 1; j < length; j++ {
			e.Data[j*length+i] = e.Data[i*length+j]
		}
	}
	return Multi{
		E: e,
		U: u,
	}
}

// LearnA factors a matrix into AA^T
func (m *Multi) LearnA(rng *rand.Rand, debug *[]float32) {
	length := m.U.Cols
	a := NewRandomMatrix(length, length)
	type Sample struct {
		Cost   float32
		Matrix Matrix
	}
	samples := make([]Sample, Samples)
	for i := 0; i < 1024; i++ {
		for j := range samples {
			sample := a.Sample(rng)
			cost := Avg(Quadratic(MulT(sample, T(sample)), m.E))
			samples[j].Matrix = sample
			samples[j].Cost = cost.Data[0]
		}
		sort.Slice(samples, func(i, j int) bool {
			return samples[i].Cost < samples[j].Cost
		})

		// https://stats.stackexchange.com/questions/6534/how-do-i-calculate-a-weighted-standard-deviation-in-excel
		weights, sum := make([]float32, Window), float32(0)
		for i := range weights {
			sum += 1 / samples[i].Cost
			weights[i] = 1 / samples[i].Cost
		}
		for i := range weights {
			weights[i] /= sum
		}

		aa := NewRandomMatrix(length, length)
		for j := range aa.Data {
			aa.Data[j].StdDev = 0
		}
		for i := range samples[:Window] {
			for j, value := range samples[i].Matrix.Data {
				aa.Data[j].Mean += weights[i] * value
			}
		}
		for i := range samples[:Window] {
			for j, value := range samples[i].Matrix.Data {
				diff := aa.Data[j].Mean - value
				aa.Data[j].StdDev += weights[i] * diff * diff
			}
		}
		for i := range aa.Data {
			aa.Data[i].StdDev /= (Window - 1.0) / Window
			aa.Data[i].StdDev = float32(math.Sqrt(float64(aa.Data[i].StdDev)))
		}
		a = aa
		if samples[0].Cost < 1e-6 {
			break
		}
	}
	m.A = samples[0].Matrix
}

// Sample samples from the multivariate distribution
func (m Multi) Sample(rng *rand.Rand) Matrix {
	length := m.U.Cols
	s := NewMatrix(0, length, 1)
	for i := 0; i < length; i++ {
		s.Data = append(s.Data, float32(rng.NormFloat64()))
	}
	return Add(MulT(m.A, s), m.U)
}
