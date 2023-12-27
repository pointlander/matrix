// Copyright 2023 The Robot Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package matrix

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/pointlander/gradient/tf32"
	"github.com/pointlander/matrix/vector"
)

const (
	// S is the scaling factor for the softmax
	S = 1.0 - 1e-300
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

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
func SelfEntropy(Q, K, V Matrix) []float32 {
	entropies, values, results := make([]float32, V.Cols), make([]float32, K.Rows), make([]float32, 0, K.Rows)
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
		softmax(entropies)

		entropy := 0.0
		for _, e := range entropies {
			entropy += float64(e) * math.Log(float64(e))
		}
		results = append(results, float32(-entropy))
	}
	return results
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

// LearnA factores a matrix into AA^T
func (m *Multi) LearnA(debug *[]float32) {
	rng := rand.New(rand.NewSource(1))
	length := m.U.Cols

	set := tf32.NewSet()
	set.Add("A", length, length)
	set.Add("E", length, length)
	set.Weights[1].X = append(set.Weights[1].X, m.E.Data...)

	for _, w := range set.Weights[:1] {
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, float32(rng.NormFloat64()*factor))
		}
	}

	deltas := make([][]float32, 0, 8)
	for _, p := range set.Weights {
		deltas = append(deltas, make([]float32, len(p.X)))
	}

	cost := tf32.Avg(tf32.Quadratic(tf32.Mul(set.Get("A"), tf32.T(set.Get("A"))), set.Get("E")))
	alpha, eta, iterations := float32(.01), float32(.01), 8*2048
	i := 0
	for i < iterations {
		total := float32(0.0)
		set.Zero()

		total += tf32.Gradient(cost).X[0]
		sum := float32(0.0)
		for _, p := range set.Weights {
			for _, d := range p.D {
				sum += d * d
			}
		}
		norm := float32(math.Sqrt(float64(sum)))
		scaling := float32(1.0)
		if norm > 1 {
			scaling = 1 / norm
		}

		w := set.Weights[0]
		for k, d := range w.D {
			deltas[0][k] = alpha*deltas[0][k] - eta*d*scaling
			set.Weights[0].X[k] += deltas[0][k]
		}

		if debug != nil {
			*debug = append(*debug, total)
		}
		i++
	}

	a := NewMatrix(0, set.Weights[0].S[0], set.Weights[0].S[1])
	for _, v := range set.Weights[0].X {
		a.Data = append(a.Data, v)
	}
	m.A = a
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
