package value

import "core:slice"
import "core:fmt"

Op :: enum u8 {
	Add,
	Sub,
	Mul,
	Pow,
	ReLU,
	Sigmoid
}


Backward :: struct {
	self: ^Value,
	other: ^Value,
	out: ^Value,

	function: proc(backward: ^Backward),
}

empty_backward :: proc() -> Backward {
	return Backward{nil, nil, nil, nil}
}


Value :: struct {
	data: f32,
	gradient: f32,

	backward: Backward,
	children: [dynamic]^Value,
	op: Op
}


new :: proc(x: f32) -> Value {
	return Value{x, 0.0, empty_backward(), make([dynamic]^Value, 0, 2), nil};
}

backward :: proc(self: ^Value) {
	topology: [dynamic]^Value;
	visited: map[^Value]b8 = map[^Value]b8{};

	build_topology :: proc(value: ^Value, topology: ^[dynamic]^Value, visited: ^(map[^Value]b8)) {
		if value not_in visited {
			visited[value] = true;

			for child in value.children {
				build_topology(child, topology, visited)
			}

			append(topology, value)
		}
	}

	build_topology(self, &topology, &visited);

	self.gradient = 1.0;

	reverse_topology := topology[:];
	slice.reverse(reverse_topology[:]);

	for value in reverse_topology {
		if value.backward.self != nil {
			value.backward.function(
					&value.backward
			);
		}
	}
}



add_backward :: proc(backward: ^Backward) {
	backward.self.gradient += backward.out.gradient;
	backward.other.gradient += backward.out.gradient;
}

add :: proc(out: ^Value, a: ^Value, b: ^Value) {
	out.data = a.data + b.data;

	out.op = .Add;
	out.backward = Backward{
		a, b, out, add_backward
	};

	append(&out.children, a, b)
}


mul_backward :: proc(backward: ^Backward) {
	backward.self.gradient += backward.other.data * backward.out.gradient;
	backward.other.gradient += backward.self.data * backward.out.gradient;
}

mul :: proc(out: ^Value, a: ^Value, b: ^Value) {
	out.data = a.data * b.data;

	out.op = .Mul;
	out.backward = Backward{
		a, b, out, mul_backward
	};

	append(&out.children, a, b)
}
