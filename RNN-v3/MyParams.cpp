#include "MyParams.h"

int MyParams::n_features = 17; // num of columns os matData
int MyParams::n_hidden = 50;	// num of hidden neurons
int MyParams::n_output_classes = 10;	// num of predicted classes of Scenarios
float MyParams::score_min = 6.0f;
float MyParams::score_max = 8.9f;

MyArray* MyParams::Wxh = MyArray::randn(MyParams::n_hidden, MyParams::n_features);
MyArray* MyParams::Whh = MyArray::randn(MyParams::n_hidden, MyParams::n_hidden);
MyArray* MyParams::Why = MyArray::randn(MyParams::n_output_classes, MyParams::n_hidden);
MyArray* MyParams::bh = MyArray::randn(MyParams::n_hidden, 1);
MyArray* MyParams::by = MyArray::randn(MyParams::n_output_classes, 1);

MyParams::MyParams()
{
}


MyParams::~MyParams()
{
}
