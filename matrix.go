// Copyright 2023 The Matrix Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package matrix

import (
	"fmt"
	"math"

	"github.com/pointlander/matrix/vector"
)

const (
	// S is the scaling factor for the softmax
	S = 1.0 - 1e-300
	// LFSRMask is a LFSR mask with a maximum period
	LFSRMask = 0x80000057
)

// Rand is a random number generator
type Rand uint32

// Uint32 returns the next random number
func (r *Rand) Uint32() uint32 {
	lfsr := *r
	lfsr = (lfsr >> 1) ^ (-(lfsr & 1) & LFSRMask)
	*r = lfsr
	return uint32(lfsr)
}

// Float64 generates a uniform float64
func (r *Rand) Float64() float64 {
	return float64(r.Uint32()) / math.MaxUint32
}

// Random is a random variable
type Random struct {
	Mean   float64
	StdDev float64
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
	factor := math.Sqrt(2.0 / float64(cols))
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
func (r RandomMatrix) Sample(rng *Rand) Generator {
	seed := rng.Uint32() + 1
	if seed == 0 {
		seed = 1
	}
	return Generator{
		Distribution: r,
		Seed:         seed,
	}
}

// SampleDiscrete generates a discrete matrix sample
func (r RandomMatrix) SampleDiscrete(rng *Rand) Matrix {
	sample := NewMatrix(r.Cols, r.Rows)
	for _, value := range r.Data {
		v := rng.NormFloat64()*value.StdDev + value.Mean
		if v > 0 {
			v = 1
		} else {
			v = -1
		}
		sample.Data = append(sample.Data, float32(v))
	}
	return sample
}

// Generator generates a matrix by sampling from a probability distribution
type Generator struct {
	Distribution RandomMatrix
	Seed         uint32
}

// Sample samples a matrix
func (g Generator) Sample() Matrix {
	rng := Rand(g.Seed)
	sample := NewMatrix(g.Distribution.Cols, g.Distribution.Rows)
	for _, v := range g.Distribution.Data {
		value := rng.NormFloat64()*v.StdDev + v.Mean
		sample.Data = append(sample.Data, float32(value))
	}
	return sample
}

// Matrix is a float32 matrix
type Matrix struct {
	Cols int
	Rows int
	Data []float32
}

// NewMatrix creates a new float32 matrix
func NewMatrix(cols, rows int, data ...float32) Matrix {
	if data == nil {
		data = make([]float32, 0, cols*rows)
	}
	return Matrix{
		Cols: cols,
		Rows: rows,
		Data: data,
	}
}

// NewZeroMatrix creates a new float32 matrix of zeros
func NewZeroMatrix(cols, rows int) Matrix {
	length := cols * rows
	return Matrix{
		Cols: cols,
		Rows: rows,
		Data: make([]float32, length, length),
	}
}

// NewIdentityMatrix creates a new float32 identity matrix of zeros
func NewIdentityMatrix(size int) Matrix {
	length := size * size
	data := make([]float32, length, length)
	for i := 0; i < size; i++ {
		data[i*size+i] = 1
	}
	return Matrix{
		Cols: size,
		Rows: size,
		Data: data,
	}
}

// NewCoord creates a new coordinate
func NewCoord(cols, rows int) Matrix {
	return Matrix{
		Cols: cols,
		Rows: rows,
	}
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
func (m Matrix) MulT(n Matrix) Matrix {
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
func (m Matrix) Mul(n Matrix) Matrix {
	return n.T().MulT(m)
}

// Add adds two float32 matrices
func (m Matrix) Add(n Matrix) Matrix {
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
func (m Matrix) Sub(n Matrix) Matrix {
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

// H is the hadamard product of two matricies
func (m Matrix) H(n Matrix) Matrix {
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
		o.Data = append(o.Data, value*n.Data[i%lenb])
	}
	return o
}

// Sigmoid computes the sigmoid of a matrix
func (m Matrix) Sigmoid() Matrix {
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
func (m Matrix) Step() Matrix {
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
func (m Matrix) Quadratic(n Matrix) Matrix {
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
func (m Matrix) QuadraticSet(n Matrix, set []int) Matrix {
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
func (m Matrix) Avg() Matrix {
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
func (m Matrix) T() Matrix {
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
func (m Matrix) Normalize() Matrix {
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

func softmax64(values []float64) {
	max := 0.0
	for _, v := range values {
		if v > max {
			max = v
		}
	}
	s := max * S
	sum := 0.0
	for j, value := range values {
		values[j] = math.Exp(value - s)
		sum += values[j]
	}
	for j, value := range values {
		values[j] = value / sum
	}
}

func dot32(a, b []float32) float64 {
	sum := 0.0
	for i, v := range a {
		sum += float64(v) * float64(b[i])
	}
	return sum
}

func dot64(a []float64, b []float32) float64 {
	sum := 0.0
	for i, v := range a {
		sum += v * float64(b[i])
	}
	return sum
}

func taylor(values []float32) {
	sum := float32(0.0)
	for j, value := range values {
		values[j] = float32(1 + value + value*value/2)
		sum += values[j]
	}
	for j, value := range values {
		values[j] = value / sum
	}
}

func taylor64(values []float64) {
	sum := 0.0
	for j, value := range values {
		values[j] = 1 + value + value*value/2
		sum += values[j]
	}
	for j, value := range values {
		values[j] = value / sum
	}
}

// SelfAttention computes the self attention of Q, K, V
func SelfAttention(Q, K, V Matrix) Matrix {
	o := Matrix{
		Cols: V.Cols,
		Rows: K.Rows,
		Data: make([]float32, 0, V.Rows*K.Rows),
	}
	outputs, values := make([]float32, V.Cols), make([]float32, Q.Rows)
	V = V.T()
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
		//softmax(outputs)
		o.Data = append(o.Data, outputs...)
	}
	return o
}

// SelfEntropy computes the self entropy of Q, K, V
func SelfEntropy(Q, K, V Matrix) []float32 {
	entropies, values, results := make([]float32, V.Cols), make([]float32, K.Rows), make([]float32, 0, K.Rows)
	V = V.T()
	for i := 0; i < K.Rows; i++ {
		K := K.Data[i*K.Cols : (i+1)*K.Cols]
		for j := 0; j < Q.Rows; j++ {
			Q := Q.Data[j*Q.Cols : (j+1)*Q.Cols]
			values[j] = vector.Dot(K, Q)
		}
		taylor(values)

		for j := 0; j < V.Rows; j++ {
			V := V.Data[j*V.Cols : (j+1)*V.Cols]
			entropies[j] = vector.Dot(values, V)
		}
		taylor(entropies)

		entropy := 0.0
		for _, e := range entropies {
			entropy += float64(e) * math.Log(float64(e))
		}
		results = append(results, float32(-entropy))
	}
	return results
}

// SelfEntropy64 computes the self entropy of Q, K, V
func SelfEntropy64(Q, K, V Matrix) []float64 {
	entropies, values, results := make([]float64, V.Cols), make([]float64, K.Rows), make([]float64, 0, K.Rows)
	V = V.T()
	for i := 0; i < K.Rows; i++ {
		K := K.Data[i*K.Cols : (i+1)*K.Cols]
		for j := 0; j < Q.Rows; j++ {
			Q := Q.Data[j*Q.Cols : (j+1)*Q.Cols]
			values[j] = dot32(K, Q)
		}
		softmax64(values)

		for j := 0; j < V.Rows; j++ {
			V := V.Data[j*V.Cols : (j+1)*V.Cols]
			entropies[j] = dot64(values, V)
		}
		softmax64(entropies)

		entropy := 0.0
		for _, e := range entropies {
			entropy += e * math.Log(e)
		}
		results = append(results, -entropy)
	}
	return results
}

// Everett is the everett activation function
func (m Matrix) Everett() Matrix {
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
func (m Matrix) TaylorSoftmax() Matrix {
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
func (m Matrix) LU() (Matrix, Matrix) {
	n := m.Cols
	lower := NewZeroMatrix(n, n)
	upper := NewZeroMatrix(n, n)
	for i := 0; i < n; i++ {
		for k := i; k < n; k++ {
			sum := float32(0.0)
			for j := 0; j < i; j++ {
				sum += lower.Data[i*n+j] * upper.Data[j*n+k]
			}
			upper.Data[i*n+k] = m.Data[i*n+k] - sum
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
					lower.Data[k*n+i] = (m.Data[k*n+i] - sum) / upper.Data[i*n+i]
				}
			}
		}
	}
	return lower, upper
}

// Determinant calculates the determinant of a matrix
func (m Matrix) Determinant() (float64, error) {
	l, u := m.LU()
	det := 1.0
	for i := 0; i < l.Cols; i++ {
		det *= float64(l.Data[i*l.Cols+i]) * float64(u.Data[i*l.Cols+i])
	}
	return det, nil
}

// Inverse computes the matrix inverse
func (m Matrix) Inverse(rng *Rand) (ai Matrix) {
	identity := NewIdentityMatrix(m.Cols)
	s := Meta(512, 1e-1, .1, rng, 4, .1, 1, false, func(samples []Sample, x ...Matrix) {
		done := make(chan bool, 8)
		process := func(index int) {
			x := samples[index].Vars[0][0].Sample()
			y := samples[index].Vars[0][1].Sample()
			z := samples[index].Vars[0][2].Sample()
			ai := x.Add(y.H(z))
			//ai := SelfAttention(x, y, z)
			cost := m.MulT(ai).Quadratic(identity).Avg()
			samples[index].Cost = float64(cost.Data[0])
			done <- true
		}
		for j := range samples {
			go process(j)
		}
		for range samples {
			<-done
		}
	}, m)
	return s.Vars[0][0].Sample().Add(s.Vars[0][1].Sample().H(s.Vars[0][2].Sample()))
	//return SelfAttention(s.Vars[0][0].Sample(), s.Vars[0][1].Sample(), s.Vars[0][2].Sample())
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
	a := NewZeroMatrix(vars, vars)
	for i := 0; i < vars; i++ {
		for j := 0; j < vars; j++ {
			if i == j {
				a.Data[i*vars+j] = factor
			}
		}
	}
	u := NewZeroMatrix(vars, 1)
	return Multi{
		A: a,
		U: u,
	}
}

// NewMultiFromData creates a new multivariate distribution
func NewMultiFromData(vars Matrix) Multi {
	length := vars.Rows
	e := NewZeroMatrix(length, length)
	u := NewZeroMatrix(length, 1)
	for i := 0; i < vars.Rows; i++ {
		for j := 0; j < vars.Cols; j++ {
			u.Data[i] += vars.Data[i*vars.Cols+j]
		}
	}
	size := vars.Cols
	for i := range u.Data {
		u.Data[i] /= float32(size)
	}
	for i := 0; i < length; i++ {
		for j := i; j < length; j++ {
			for k := 0; k < size; k++ {
				e.Data[i*length+j] += (vars.Data[i*vars.Cols+k] - u.Data[i]) *
					(vars.Data[j*vars.Cols+k] - u.Data[j])
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
func (m *Multi) LearnA(rng *Rand, debug *[]float32) {
	optimizer := NewOptimizer(rng, 14, .1, 1, func(samples []Sample, x ...Matrix) {
		done := make(chan bool, 8)
		process := func(index int) {
			x := samples[index].Vars[0][0].Sample()
			y := samples[index].Vars[0][1].Sample()
			z := samples[index].Vars[0][2].Sample()
			sample := x.Add(y.H(z))
			cost := sample.MulT(sample.T()).Quadratic(m.E).Avg()
			samples[index].Cost = float64(cost.Data[0])
			done <- true
		}
		for j := range samples {
			go process(j)
		}
		for range samples {
			<-done
		}
	}, m.E)
	s := optimizer.Optimize(1e-6)
	m.A = s.Vars[0][0].Sample().Add(s.Vars[0][1].Sample().H(s.Vars[0][2].Sample()))
}

// Sample samples from the multivariate distribution
func (m Multi) Sample(rng *Rand) Matrix {
	length := m.U.Cols
	s := NewMatrix(length, 1)
	for i := 0; i < length; i++ {
		s.Data = append(s.Data, float32(rng.NormFloat64()))
	}
	return m.A.MulT(s).Add(m.U)
}

// LinearRegression computes linear regression
// https://www.geeksforgeeks.org/linear-regression-python-implementation/
func LinearRegression(x, y Matrix) (b0, b1 float64) {
	n := float64(len(x.Data))
	xu := 0.0
	for _, value := range x.Data {
		xu += float64(value)
	}
	xu /= n
	yu := 0.0
	for _, value := range y.Data {
		yu += float64(value)
	}
	yu /= n
	ssxx, ssxy := 0.0, 0.0
	for i, value := range x.Data {
		/*diffx := float64(value) - xu
		ssxx += diffx * diffx
		diffy := float64(y.Data[i]) - yu
		ssxy += diffx * diffy*/
		ssxy += float64(y.Data[i]) * float64(value)
		ssxx += float64(value) * float64(value)
	}
	ssxy -= n * yu * xu
	ssxx -= n * xu * xu
	b1 = ssxy / ssxx
	b0 = yu - b1*xu
	return b0, b1
}
