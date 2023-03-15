package linear_models

/*
#include "linear.h"
#include <stdlib.h>
*/
import "C"
import "fmt"
import "unsafe"
import "math"

type Problem struct {
	c_prob C.struct_problem
	Y      []float64
	X      [][]C.struct_feature_node
}

type Parameter struct {
	c_param      C.struct_parameter
	WeightLabel  []int32
	Weight       []float64
	cWeightLabel *C.int
	cWeight      *C.double
}

type Model struct {
	c_model unsafe.Pointer
}

const (
	L2R_LR              = C.L2R_LR
	L2R_L2LOSS_SVC_DUAL = C.L2R_L2LOSS_SVC_DUAL
	L2R_L2LOSS_SVC      = C.L2R_L2LOSS_SVC
	L2R_L1LOSS_SVC_DUAL = C.L2R_L1LOSS_SVC_DUAL
	MCSVM_CS            = C.MCSVM_CS
	L1R_L2LOSS_SVC      = C.L1R_L2LOSS_SVC
	L1R_LR              = C.L1R_LR
	L2R_LR_DUAL         = C.L2R_LR_DUAL
)

func NewParameter(solver_type int, C float64, eps float64) *Parameter {
	param := Parameter{}
	param.c_param.solver_type = C.int(solver_type)
	param.c_param.eps = C.double(eps)
	param.c_param.C = C.double(C)
	param.c_param.nr_weight = C.int(0)
	param.c_param.weight_label = nil
	param.c_param.weight = nil

	return &param
}

func NewProblem(X [][]float64, y []float64, bias float64) *Problem {
	prob := Problem{}
	prob.c_prob.l = C.int(len(X))
	prob.c_prob.n = C.int(len(X[0]) + 1)

	prob.c_prob.x = convert_features(X, bias)
	c_y := make([]C.double, len(y))
	for i := 0; i < len(y); i++ {
		c_y[i] = C.double(y[i])
	}
	prob.c_prob.y = &c_y[0]
	prob.c_prob.bias = C.double(-1)

	// Set the Y and X fields of the Problem struct
	prob.Y = y
	prob.X = make([][]C.struct_feature_node, len(X))
	for i := 0; i < len(X); i++ {
		prob.X[i] = make([]C.struct_feature_node, len(X[i])+1) // +1 for the bias
		for j := 0; j < len(X[i]); j++ {
			prob.X[i][j].index = C.int(j + 1)
			prob.X[i][j].value = C.double(X[i][j])
		}
		// Add the bias
		prob.X[i][len(X[i])].index = C.int(-1)
		prob.X[i][len(X[i])].value = C.double(bias)
	}

	return &prob
}

func Train(prob *Problem, param *Parameter) *Model {
	if !validateInputData(prob) {
		panic("Invalid input data: found NaN or Inf values.")
	}
	libLinearHookPrintFunc() // Sets up logging

	tmpCProb := C.struct_problem{
		l:    C.int(prob.c_prob.l),
		n:    C.int(prob.c_prob.n),
		y:    (*C.double)(unsafe.Pointer(prob.c_prob.y)),
		x:    (**C.struct_feature_node)(unsafe.Pointer(prob.c_prob.x)),
		bias: C.double(prob.c_prob.bias),
	}

	// Allocate memory on the C side
	fmt.Println("Allocating memory for c_prob and c_param")
	c_prob := (*C.struct_problem)(C.malloc(C.size_t(unsafe.Sizeof(tmpCProb))))
	c_param := (*C.struct_parameter)(C.malloc(C.size_t(unsafe.Sizeof(param.c_param))))

	// Copy the content of Go pointers to the newly allocated C memory
	*c_prob = tmpCProb
	*c_param = param.c_param

	// Add fmt print statement before the C function call
	fmt.Println("Before calling C.train")

	// Call the C function with the C pointers
	modelPtr := C.train(c_prob, c_param)

	// Add fmt print statement after the C function call
	fmt.Println("After calling C.train")

	// Free the allocated memory on the C side
	fmt.Println("Freeing memory for c_prob and c_param")
	C.free(unsafe.Pointer(c_prob))
	C.free(unsafe.Pointer(c_param))

	return &Model{unsafe.Pointer(modelPtr)}
}

func Export(model *Model, filePath string) error {
	status := C.save_model(C.CString(filePath), (*C.struct_model)(model.c_model))
	if status != 0 {
		return fmt.Errorf("Problem occured during export to %s (status was %d)", filePath, status)
	}
	return nil
}

func Load(model *Model, filePath string) error {
	model.c_model = unsafe.Pointer(C.load_model(C.CString(filePath)))
	if model.c_model == nil {
		return fmt.Errorf("Something went wrong")
	}
	return nil
}

func Predict(model *Model, x []float64) float64 {
	c_x := convert_vector(x, 0)
	c_y := C.predict((*C.struct_model)(model.c_model), c_x)
	y := float64(c_y)
	return y
}
func convert_vector(x []float64, bias float64) *C.struct_feature_node {
	n_ele := 0
	for i := 0; i < len(x); i++ {
		if x[i] > 0 {
			n_ele++
		}
	}
	n_ele += 2

	c_x := make([]C.struct_feature_node, n_ele)
	j := 0
	for i := 0; i < len(x); i++ {
		if x[i] > 0 {
			c_x[j].index = C.int(i + 1)
			c_x[j].value = C.double(x[i])
			j++
		}
	}
	if bias > 0 {
		c_x[j].index = C.int(0)
		c_x[j].value = C.double(0)
		j++
	}
	c_x[j].index = C.int(-1)
	return &c_x[0]
}
func convert_features(X [][]float64, bias float64) **C.struct_feature_node {
	n_samples := len(X)
	n_elements := 0

	for i := 0; i < n_samples; i++ {
		for j := 0; j < len(X[i]); j++ {
			if X[i][j] != 0.0 {
				n_elements++
			}
			n_elements++ //for bias
		}
	}

	x_space := make([]C.struct_feature_node, n_elements+n_samples)

	cursor := 0
	x := make([]*C.struct_feature_node, n_samples)
	var c_x **C.struct_feature_node

	for i := 0; i < n_samples; i++ {
		x[i] = &x_space[cursor]

		for j := 0; j < len(X[i]); j++ {
			if X[i][j] != 0.0 {
				x_space[cursor].index = C.int(j + 1)
				x_space[cursor].value = C.double(X[i][j])
				cursor++
			}
			if bias > 0 {
				x_space[cursor].index = C.int(0)
				x_space[cursor].value = C.double(bias)
				cursor++
			}
		}
		x_space[cursor].index = C.int(-1)
		cursor++
	}
	c_x = &x[0]
	return c_x
}

func isValidNumber(x float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0)
}

func validateInputData(prob *Problem) bool {
	for i := 0; i < len(prob.Y); i++ {
		if !isValidNumber(prob.Y[i]) {
			return false
		}
	}

	for _, x := range prob.X {
		for j := 0; j < len(x); j++ {
			if !isValidNumber(float64(x[j].index)) || !isValidNumber(float64(x[j].value)) {
				return false
			}
		}
	}

	return true
}
