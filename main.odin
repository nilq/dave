package main

import "core:fmt"
import "value"

main :: proc() {
	// f(x) = 2x

	a := value.new(2.0);
	b := value.new(3.0);

	c := value.new(1.0);

	value.mul(&c, &a, &b);
	value.backward(&c);

	fmt.printf("%#v", c)
}
