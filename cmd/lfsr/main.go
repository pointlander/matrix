// Copyright 2023 The Matrix Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
)

func searchLFSR32() {
	count, polynomial := 0, uint32(0x80000000)
	for polynomial != 0 {
		lfsr, period := uint32(1), 0
		for {
			lfsr = (lfsr >> 1) ^ (-(lfsr & 1) & polynomial)
			period++
			if lfsr == 1 {
				break
			}
		}
		fmt.Printf("%v period=%v\n", count, period)
		if period == math.MaxUint32 {
			fmt.Printf("%x\n", polynomial)
			return
		}
		count++
		polynomial++
	}
}

func main() {
	searchLFSR32()
}
