ş˙

 ď
:
Add
x"T
y"T
z"T"
Ttype:
2	
A
AddV2
x"T
y"T
z"T"
Ttype:
2	
h
All	
input

reduction_indices"Tidx

output
"
	keep_dimsbool( "
Tidxtype0:
2	
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeint
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp

OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint˙˙˙˙˙˙˙˙˙"	
Ttype"
TItype0	:
2	
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
6
Pow
x"T
y"T
z"T"
Ttype:

2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
Ľ
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
@
StaticRegexFullMatch	
input

output
"
patternstring
2
StopGradient

input"T
output"T"	
Ttype
ö
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2

TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape
9
VarIsInitializedOp
resource
is_initialized
"serve*1.15.02unknown8ëŻ	

global_step/Initializer/zerosConst*
_class
loc:@global_step*
_output_shapes
: *
dtype0	*
value	B	 R 

global_stepVarHandleOp*
_class
loc:@global_step*
_output_shapes
: *
dtype0	*
shape: *
shared_nameglobal_step
g
,global_step/IsInitialized/VarIsInitializedOpVarIsInitializedOpglobal_step*
_output_shapes
: 
_
global_step/AssignAssignVariableOpglobal_stepglobal_step/Initializer/zeros*
dtype0	
c
global_step/Read/ReadVariableOpReadVariableOpglobal_step*
_output_shapes
: *
dtype0	
p
PlaceholderPlaceholder*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0	*
shape:˙˙˙˙˙˙˙˙˙
r
Placeholder_1Placeholder*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0	*
shape:˙˙˙˙˙˙˙˙˙
r
Placeholder_2Placeholder*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0	*
shape:˙˙˙˙˙˙˙˙˙
@
ShapeShapePlaceholder*
T0	*
_output_shapes
:
]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
­
strided_sliceStridedSliceShapestrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
f
zeros/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙
a
zeros/ReshapeReshapestrided_slicezeros/Reshape/shape*
T0*
_output_shapes
:
P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
W
zerosFillzeros/Reshapezeros/Const*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
B
Shape_1ShapePlaceholder*
T0	*
_output_shapes
:
_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
ˇ
strided_slice_1StridedSliceShape_1strided_slice_1/stackstrided_slice_1/stack_1strided_slice_1/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
q
bert/embeddings/ExpandDims/dimConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙

bert/embeddings/ExpandDims
ExpandDimsPlaceholderbert/embeddings/ExpandDims/dim*
T0	*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ç
Bbert/embeddings/word_embeddings/Initializer/truncated_normal/shapeConst*2
_class(
&$loc:@bert/embeddings/word_embeddings*
_output_shapes
:*
dtype0*
valueB":w     
ş
Abert/embeddings/word_embeddings/Initializer/truncated_normal/meanConst*2
_class(
&$loc:@bert/embeddings/word_embeddings*
_output_shapes
: *
dtype0*
valueB
 *    
ź
Cbert/embeddings/word_embeddings/Initializer/truncated_normal/stddevConst*2
_class(
&$loc:@bert/embeddings/word_embeddings*
_output_shapes
: *
dtype0*
valueB
 *
×Ł<

Lbert/embeddings/word_embeddings/Initializer/truncated_normal/TruncatedNormalTruncatedNormalBbert/embeddings/word_embeddings/Initializer/truncated_normal/shape*
T0*2
_class(
&$loc:@bert/embeddings/word_embeddings*!
_output_shapes
:şî*
dtype0*
seed˛
ş
@bert/embeddings/word_embeddings/Initializer/truncated_normal/mulMulLbert/embeddings/word_embeddings/Initializer/truncated_normal/TruncatedNormalCbert/embeddings/word_embeddings/Initializer/truncated_normal/stddev*
T0*2
_class(
&$loc:@bert/embeddings/word_embeddings*!
_output_shapes
:şî
¨
<bert/embeddings/word_embeddings/Initializer/truncated_normalAdd@bert/embeddings/word_embeddings/Initializer/truncated_normal/mulAbert/embeddings/word_embeddings/Initializer/truncated_normal/mean*
T0*2
_class(
&$loc:@bert/embeddings/word_embeddings*!
_output_shapes
:şî
Ń
bert/embeddings/word_embeddingsVarHandleOp*2
_class(
&$loc:@bert/embeddings/word_embeddings*
_output_shapes
: *
dtype0*
shape:şî*0
shared_name!bert/embeddings/word_embeddings

@bert/embeddings/word_embeddings/IsInitialized/VarIsInitializedOpVarIsInitializedOpbert/embeddings/word_embeddings*
_output_shapes
: 
Ś
&bert/embeddings/word_embeddings/AssignAssignVariableOpbert/embeddings/word_embeddings<bert/embeddings/word_embeddings/Initializer/truncated_normal*
dtype0

3bert/embeddings/word_embeddings/Read/ReadVariableOpReadVariableOpbert/embeddings/word_embeddings*!
_output_shapes
:şî*
dtype0
ó
 bert/embeddings/embedding_lookupResourceGatherbert/embeddings/word_embeddingsbert/embeddings/ExpandDims*
Tindices0	*2
_class(
&$loc:@bert/embeddings/word_embeddings*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0
Ç
)bert/embeddings/embedding_lookup/IdentityIdentity bert/embeddings/embedding_lookup*
T0*2
_class(
&$loc:@bert/embeddings/word_embeddings*1
_output_shapes
:˙˙˙˙˙˙˙˙˙

+bert/embeddings/embedding_lookup/Identity_1Identity)bert/embeddings/embedding_lookup/Identity*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
_
bert/embeddings/ShapeShapebert/embeddings/ExpandDims*
T0	*
_output_shapes
:
m
#bert/embeddings/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
o
%bert/embeddings/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
o
%bert/embeddings/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
ý
bert/embeddings/strided_sliceStridedSlicebert/embeddings/Shape#bert/embeddings/strided_slice/stack%bert/embeddings/strided_slice/stack_1%bert/embeddings/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
b
bert/embeddings/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :
b
bert/embeddings/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :
´
bert/embeddings/Reshape/shapePackbert/embeddings/strided_slicebert/embeddings/Reshape/shape/1bert/embeddings/Reshape/shape/2*
N*
T0*
_output_shapes
:
Ś
bert/embeddings/ReshapeReshape+bert/embeddings/embedding_lookup/Identity_1bert/embeddings/Reshape/shape*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
^
bert/embeddings/Shape_1Shapebert/embeddings/Reshape*
T0*
_output_shapes
:
o
%bert/embeddings/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
q
'bert/embeddings/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
q
'bert/embeddings/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:

bert/embeddings/strided_slice_1StridedSlicebert/embeddings/Shape_1%bert/embeddings/strided_slice_1/stack'bert/embeddings/strided_slice_1/stack_1'bert/embeddings/strided_slice_1/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Ó
Hbert/embeddings/token_type_embeddings/Initializer/truncated_normal/shapeConst*8
_class.
,*loc:@bert/embeddings/token_type_embeddings*
_output_shapes
:*
dtype0*
valueB"      
Ć
Gbert/embeddings/token_type_embeddings/Initializer/truncated_normal/meanConst*8
_class.
,*loc:@bert/embeddings/token_type_embeddings*
_output_shapes
: *
dtype0*
valueB
 *    
Č
Ibert/embeddings/token_type_embeddings/Initializer/truncated_normal/stddevConst*8
_class.
,*loc:@bert/embeddings/token_type_embeddings*
_output_shapes
: *
dtype0*
valueB
 *
×Ł<
ť
Rbert/embeddings/token_type_embeddings/Initializer/truncated_normal/TruncatedNormalTruncatedNormalHbert/embeddings/token_type_embeddings/Initializer/truncated_normal/shape*
T0*8
_class.
,*loc:@bert/embeddings/token_type_embeddings*
_output_shapes
:	*
dtype0*
seed˛*
seed2
Đ
Fbert/embeddings/token_type_embeddings/Initializer/truncated_normal/mulMulRbert/embeddings/token_type_embeddings/Initializer/truncated_normal/TruncatedNormalIbert/embeddings/token_type_embeddings/Initializer/truncated_normal/stddev*
T0*8
_class.
,*loc:@bert/embeddings/token_type_embeddings*
_output_shapes
:	
ž
Bbert/embeddings/token_type_embeddings/Initializer/truncated_normalAddFbert/embeddings/token_type_embeddings/Initializer/truncated_normal/mulGbert/embeddings/token_type_embeddings/Initializer/truncated_normal/mean*
T0*8
_class.
,*loc:@bert/embeddings/token_type_embeddings*
_output_shapes
:	
á
%bert/embeddings/token_type_embeddingsVarHandleOp*8
_class.
,*loc:@bert/embeddings/token_type_embeddings*
_output_shapes
: *
dtype0*
shape:	*6
shared_name'%bert/embeddings/token_type_embeddings

Fbert/embeddings/token_type_embeddings/IsInitialized/VarIsInitializedOpVarIsInitializedOp%bert/embeddings/token_type_embeddings*
_output_shapes
: 
¸
,bert/embeddings/token_type_embeddings/AssignAssignVariableOp%bert/embeddings/token_type_embeddingsBbert/embeddings/token_type_embeddings/Initializer/truncated_normal*
dtype0
 
9bert/embeddings/token_type_embeddings/Read/ReadVariableOpReadVariableOp%bert/embeddings/token_type_embeddings*
_output_shapes
:	*
dtype0
r
bert/embeddings/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
˙˙˙˙˙˙˙˙˙

bert/embeddings/Reshape_1ReshapePlaceholder_2bert/embeddings/Reshape_1/shape*
T0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
e
 bert/embeddings/one_hot/on_valueConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
f
!bert/embeddings/one_hot/off_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    
_
bert/embeddings/one_hot/depthConst*
_output_shapes
: *
dtype0*
value	B :
Ň
bert/embeddings/one_hotOneHotbert/embeddings/Reshape_1bert/embeddings/one_hot/depth bert/embeddings/one_hot/on_value!bert/embeddings/one_hot/off_value*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

%bert/embeddings/MatMul/ReadVariableOpReadVariableOp%bert/embeddings/token_type_embeddings*
_output_shapes
:	*
dtype0

bert/embeddings/MatMulMatMulbert/embeddings/one_hot%bert/embeddings/MatMul/ReadVariableOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
d
!bert/embeddings/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value
B :
d
!bert/embeddings/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value
B :
ź
bert/embeddings/Reshape_2/shapePackbert/embeddings/strided_slice_1!bert/embeddings/Reshape_2/shape/1!bert/embeddings/Reshape_2/shape/2*
N*
T0*
_output_shapes
:

bert/embeddings/Reshape_2Reshapebert/embeddings/MatMulbert/embeddings/Reshape_2/shape*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙

bert/embeddings/addAddV2bert/embeddings/Reshapebert/embeddings/Reshape_2*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
f
#bert/embeddings/assert_less_equal/xConst*
_output_shapes
: *
dtype0*
value
B :
f
#bert/embeddings/assert_less_equal/yConst*
_output_shapes
: *
dtype0*
value
B :
Ł
+bert/embeddings/assert_less_equal/LessEqual	LessEqual#bert/embeddings/assert_less_equal/x#bert/embeddings/assert_less_equal/y*
T0*
_output_shapes
: 
j
'bert/embeddings/assert_less_equal/ConstConst*
_output_shapes
: *
dtype0*
valueB 

%bert/embeddings/assert_less_equal/AllAll+bert/embeddings/assert_less_equal/LessEqual'bert/embeddings/assert_less_equal/Const*
_output_shapes
: 

.bert/embeddings/assert_less_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x <= y did not hold element-wise:

0bert/embeddings/assert_less_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (bert/embeddings/assert_less_equal/x:0) = 

0bert/embeddings/assert_less_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,y (bert/embeddings/assert_less_equal/y:0) = 
˘
6bert/embeddings/assert_less_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x <= y did not hold element-wise:
Ł
6bert/embeddings/assert_less_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (bert/embeddings/assert_less_equal/x:0) = 
Ł
6bert/embeddings/assert_less_equal/Assert/Assert/data_3Const*
_output_shapes
: *
dtype0*=
value4B2 B,y (bert/embeddings/assert_less_equal/y:0) = 
â
/bert/embeddings/assert_less_equal/Assert/AssertAssert%bert/embeddings/assert_less_equal/All6bert/embeddings/assert_less_equal/Assert/Assert/data_06bert/embeddings/assert_less_equal/Assert/Assert/data_1#bert/embeddings/assert_less_equal/x6bert/embeddings/assert_less_equal/Assert/Assert/data_3#bert/embeddings/assert_less_equal/y*
T	
2
Ď
Fbert/embeddings/position_embeddings/Initializer/truncated_normal/shapeConst*6
_class,
*(loc:@bert/embeddings/position_embeddings*
_output_shapes
:*
dtype0*
valueB"      
Â
Ebert/embeddings/position_embeddings/Initializer/truncated_normal/meanConst*6
_class,
*(loc:@bert/embeddings/position_embeddings*
_output_shapes
: *
dtype0*
valueB
 *    
Ä
Gbert/embeddings/position_embeddings/Initializer/truncated_normal/stddevConst*6
_class,
*(loc:@bert/embeddings/position_embeddings*
_output_shapes
: *
dtype0*
valueB
 *
×Ł<
ś
Pbert/embeddings/position_embeddings/Initializer/truncated_normal/TruncatedNormalTruncatedNormalFbert/embeddings/position_embeddings/Initializer/truncated_normal/shape*
T0*6
_class,
*(loc:@bert/embeddings/position_embeddings* 
_output_shapes
:
*
dtype0*
seed˛*
seed2
É
Dbert/embeddings/position_embeddings/Initializer/truncated_normal/mulMulPbert/embeddings/position_embeddings/Initializer/truncated_normal/TruncatedNormalGbert/embeddings/position_embeddings/Initializer/truncated_normal/stddev*
T0*6
_class,
*(loc:@bert/embeddings/position_embeddings* 
_output_shapes
:

ˇ
@bert/embeddings/position_embeddings/Initializer/truncated_normalAddDbert/embeddings/position_embeddings/Initializer/truncated_normal/mulEbert/embeddings/position_embeddings/Initializer/truncated_normal/mean*
T0*6
_class,
*(loc:@bert/embeddings/position_embeddings* 
_output_shapes
:

Ü
#bert/embeddings/position_embeddingsVarHandleOp*6
_class,
*(loc:@bert/embeddings/position_embeddings*
_output_shapes
: *
dtype0*
shape:
*4
shared_name%#bert/embeddings/position_embeddings

Dbert/embeddings/position_embeddings/IsInitialized/VarIsInitializedOpVarIsInitializedOp#bert/embeddings/position_embeddings*
_output_shapes
: 
˛
*bert/embeddings/position_embeddings/AssignAssignVariableOp#bert/embeddings/position_embeddings@bert/embeddings/position_embeddings/Initializer/truncated_normal*
dtype0

7bert/embeddings/position_embeddings/Read/ReadVariableOpReadVariableOp#bert/embeddings/position_embeddings* 
_output_shapes
:
*
dtype0
ź
$bert/embeddings/Slice/ReadVariableOpReadVariableOp#bert/embeddings/position_embeddings0^bert/embeddings/assert_less_equal/Assert/Assert* 
_output_shapes
:
*
dtype0

bert/embeddings/Slice/beginConst0^bert/embeddings/assert_less_equal/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"        

bert/embeddings/Slice/sizeConst0^bert/embeddings/assert_less_equal/Assert/Assert*
_output_shapes
:*
dtype0*
valueB"   ˙˙˙˙
ľ
bert/embeddings/SliceSlice$bert/embeddings/Slice/ReadVariableOpbert/embeddings/Slice/beginbert/embeddings/Slice/size*
Index0*
T0* 
_output_shapes
:

Ś
bert/embeddings/Reshape_3/shapeConst0^bert/embeddings/assert_less_equal/Assert/Assert*
_output_shapes
:*
dtype0*!
valueB"         

bert/embeddings/Reshape_3Reshapebert/embeddings/Slicebert/embeddings/Reshape_3/shape*
T0*$
_output_shapes
:

bert/embeddings/add_1AddV2bert/embeddings/addbert/embeddings/Reshape_3*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
˛
0bert/embeddings/LayerNorm/beta/Initializer/zerosConst*1
_class'
%#loc:@bert/embeddings/LayerNorm/beta*
_output_shapes	
:*
dtype0*
valueB*    
Č
bert/embeddings/LayerNorm/betaVarHandleOp*1
_class'
%#loc:@bert/embeddings/LayerNorm/beta*
_output_shapes
: *
dtype0*
shape:*/
shared_name bert/embeddings/LayerNorm/beta

?bert/embeddings/LayerNorm/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOpbert/embeddings/LayerNorm/beta*
_output_shapes
: 

%bert/embeddings/LayerNorm/beta/AssignAssignVariableOpbert/embeddings/LayerNorm/beta0bert/embeddings/LayerNorm/beta/Initializer/zeros*
dtype0

2bert/embeddings/LayerNorm/beta/Read/ReadVariableOpReadVariableOpbert/embeddings/LayerNorm/beta*
_output_shapes	
:*
dtype0
ł
0bert/embeddings/LayerNorm/gamma/Initializer/onesConst*2
_class(
&$loc:@bert/embeddings/LayerNorm/gamma*
_output_shapes	
:*
dtype0*
valueB*  ?
Ë
bert/embeddings/LayerNorm/gammaVarHandleOp*2
_class(
&$loc:@bert/embeddings/LayerNorm/gamma*
_output_shapes
: *
dtype0*
shape:*0
shared_name!bert/embeddings/LayerNorm/gamma

@bert/embeddings/LayerNorm/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOpbert/embeddings/LayerNorm/gamma*
_output_shapes
: 

&bert/embeddings/LayerNorm/gamma/AssignAssignVariableOpbert/embeddings/LayerNorm/gamma0bert/embeddings/LayerNorm/gamma/Initializer/ones*
dtype0

3bert/embeddings/LayerNorm/gamma/Read/ReadVariableOpReadVariableOpbert/embeddings/LayerNorm/gamma*
_output_shapes	
:*
dtype0

8bert/embeddings/LayerNorm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
Ç
&bert/embeddings/LayerNorm/moments/meanMeanbert/embeddings/add_18bert/embeddings/LayerNorm/moments/mean/reduction_indices*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(

.bert/embeddings/LayerNorm/moments/StopGradientStopGradient&bert/embeddings/LayerNorm/moments/mean*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ç
3bert/embeddings/LayerNorm/moments/SquaredDifferenceSquaredDifferencebert/embeddings/add_1.bert/embeddings/LayerNorm/moments/StopGradient*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙

<bert/embeddings/LayerNorm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
í
*bert/embeddings/LayerNorm/moments/varianceMean3bert/embeddings/LayerNorm/moments/SquaredDifference<bert/embeddings/LayerNorm/moments/variance/reduction_indices*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(
n
)bert/embeddings/LayerNorm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ěź+
ž
'bert/embeddings/LayerNorm/batchnorm/addAddV2*bert/embeddings/LayerNorm/moments/variance)bert/embeddings/LayerNorm/batchnorm/add/y*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙

)bert/embeddings/LayerNorm/batchnorm/RsqrtRsqrt'bert/embeddings/LayerNorm/batchnorm/add*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙

6bert/embeddings/LayerNorm/batchnorm/mul/ReadVariableOpReadVariableOpbert/embeddings/LayerNorm/gamma*
_output_shapes	
:*
dtype0
É
'bert/embeddings/LayerNorm/batchnorm/mulMul)bert/embeddings/LayerNorm/batchnorm/Rsqrt6bert/embeddings/LayerNorm/batchnorm/mul/ReadVariableOp*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
¨
)bert/embeddings/LayerNorm/batchnorm/mul_1Mulbert/embeddings/add_1'bert/embeddings/LayerNorm/batchnorm/mul*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
š
)bert/embeddings/LayerNorm/batchnorm/mul_2Mul&bert/embeddings/LayerNorm/moments/mean'bert/embeddings/LayerNorm/batchnorm/mul*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙

2bert/embeddings/LayerNorm/batchnorm/ReadVariableOpReadVariableOpbert/embeddings/LayerNorm/beta*
_output_shapes	
:*
dtype0
Ĺ
'bert/embeddings/LayerNorm/batchnorm/subSub2bert/embeddings/LayerNorm/batchnorm/ReadVariableOp)bert/embeddings/LayerNorm/batchnorm/mul_2*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
)bert/embeddings/LayerNorm/batchnorm/add_1AddV2)bert/embeddings/LayerNorm/batchnorm/mul_1'bert/embeddings/LayerNorm/batchnorm/sub*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
M
bert/encoder/ShapeShapePlaceholder*
T0	*
_output_shapes
:
j
 bert/encoder/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
l
"bert/encoder/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
l
"bert/encoder/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
î
bert/encoder/strided_sliceStridedSlicebert/encoder/Shape bert/encoder/strided_slice/stack"bert/encoder/strided_slice/stack_1"bert/encoder/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Q
bert/encoder/Shape_1ShapePlaceholder_1*
T0	*
_output_shapes
:
l
"bert/encoder/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
n
$bert/encoder/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
n
$bert/encoder/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
ř
bert/encoder/strided_slice_1StridedSlicebert/encoder/Shape_1"bert/encoder/strided_slice_1/stack$bert/encoder/strided_slice_1/stack_1$bert/encoder/strided_slice_1/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
^
bert/encoder/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
_
bert/encoder/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :
¨
bert/encoder/Reshape/shapePackbert/encoder/strided_slicebert/encoder/Reshape/shape/1bert/encoder/Reshape/shape/2*
N*
T0*
_output_shapes
:

bert/encoder/ReshapeReshapePlaceholder_1bert/encoder/Reshape/shape*
T0	*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
u
bert/encoder/CastCastbert/encoder/Reshape*

DstT0*

SrcT0	*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
Z
bert/encoder/ones/mul/yConst*
_output_shapes
: *
dtype0*
value
B :
r
bert/encoder/ones/mulMulbert/encoder/strided_slicebert/encoder/ones/mul/y*
T0*
_output_shapes
: 
[
bert/encoder/ones/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :
q
bert/encoder/ones/mul_1Mulbert/encoder/ones/mulbert/encoder/ones/mul_1/y*
T0*
_output_shapes
: 
[
bert/encoder/ones/Less/yConst*
_output_shapes
: *
dtype0*
value
B :č
r
bert/encoder/ones/LessLessbert/encoder/ones/mul_1bert/encoder/ones/Less/y*
T0*
_output_shapes
: 
]
bert/encoder/ones/packed/1Const*
_output_shapes
: *
dtype0*
value
B :
\
bert/encoder/ones/packed/2Const*
_output_shapes
: *
dtype0*
value	B :
˘
bert/encoder/ones/packedPackbert/encoder/strided_slicebert/encoder/ones/packed/1bert/encoder/ones/packed/2*
N*
T0*
_output_shapes
:
\
bert/encoder/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?

bert/encoder/onesFillbert/encoder/ones/packedbert/encoder/ones/Const*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
u
bert/encoder/mulMulbert/encoder/onesbert/encoder/Cast*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
m
bert/encoder/Shape_2Shape)bert/embeddings/LayerNorm/batchnorm/add_1*
T0*
_output_shapes
:
l
"bert/encoder/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
n
$bert/encoder/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
n
$bert/encoder/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
ř
bert/encoder/strided_slice_2StridedSlicebert/encoder/Shape_2"bert/encoder/strided_slice_2/stack$bert/encoder/strided_slice_2/stack_1$bert/encoder/strided_slice_2/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

)bert/encoder/layer_0/attention/self/ShapeShape)bert/embeddings/LayerNorm/batchnorm/add_1*
T0*
_output_shapes
:

7bert/encoder/layer_0/attention/self/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 

9bert/encoder/layer_0/attention/self/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:

9bert/encoder/layer_0/attention/self/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
á
1bert/encoder/layer_0/attention/self/strided_sliceStridedSlice)bert/encoder/layer_0/attention/self/Shape7bert/encoder/layer_0/attention/self/strided_slice/stack9bert/encoder/layer_0/attention/self/strided_slice/stack_19bert/encoder/layer_0/attention/self/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

+bert/encoder/layer_0/attention/self/Shape_1Shape)bert/embeddings/LayerNorm/batchnorm/add_1*
T0*
_output_shapes
:

9bert/encoder/layer_0/attention/self/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 

;bert/encoder/layer_0/attention/self/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:

;bert/encoder/layer_0/attention/self/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
ë
3bert/encoder/layer_0/attention/self/strided_slice_1StridedSlice+bert/encoder/layer_0/attention/self/Shape_19bert/encoder/layer_0/attention/self/strided_slice_1/stack;bert/encoder/layer_0/attention/self/strided_slice_1/stack_1;bert/encoder/layer_0/attention/self/strided_slice_1/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

+bert/encoder/layer_0/attention/self/Shape_2Shape)bert/embeddings/LayerNorm/batchnorm/add_1*
T0*
_output_shapes
:

9bert/encoder/layer_0/attention/self/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 

;bert/encoder/layer_0/attention/self/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:

;bert/encoder/layer_0/attention/self/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
ë
3bert/encoder/layer_0/attention/self/strided_slice_2StridedSlice+bert/encoder/layer_0/attention/self/Shape_29bert/encoder/layer_0/attention/self/strided_slice_2/stack;bert/encoder/layer_0/attention/self/strided_slice_2/stack_1;bert/encoder/layer_0/attention/self/strided_slice_2/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
é
Sbert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal/shapeConst*C
_class9
75loc:@bert/encoder/layer_0/attention/self/query/kernel*
_output_shapes
:*
dtype0*
valueB"      
Ü
Rbert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal/meanConst*C
_class9
75loc:@bert/encoder/layer_0/attention/self/query/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
Ţ
Tbert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal/stddevConst*C
_class9
75loc:@bert/encoder/layer_0/attention/self/query/kernel*
_output_shapes
: *
dtype0*
valueB
 *
×Ł<
Ý
]bert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal/shape*
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/self/query/kernel* 
_output_shapes
:
*
dtype0*
seed˛*
seed2
ý
Qbert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/self/query/kernel* 
_output_shapes
:

ë
Mbert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normalAddQbert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/self/query/kernel* 
_output_shapes
:


0bert/encoder/layer_0/attention/self/query/kernelVarHandleOp*C
_class9
75loc:@bert/encoder/layer_0/attention/self/query/kernel*
_output_shapes
: *
dtype0*
shape:
*A
shared_name20bert/encoder/layer_0/attention/self/query/kernel
ą
Qbert/encoder/layer_0/attention/self/query/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp0bert/encoder/layer_0/attention/self/query/kernel*
_output_shapes
: 
Ů
7bert/encoder/layer_0/attention/self/query/kernel/AssignAssignVariableOp0bert/encoder/layer_0/attention/self/query/kernelMbert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal*
dtype0
ˇ
Dbert/encoder/layer_0/attention/self/query/kernel/Read/ReadVariableOpReadVariableOp0bert/encoder/layer_0/attention/self/query/kernel* 
_output_shapes
:
*
dtype0
ł
@bert/encoder/layer_0/attention/self/query/Reshape/ReadVariableOpReadVariableOp0bert/encoder/layer_0/attention/self/query/kernel* 
_output_shapes
:
*
dtype0

7bert/encoder/layer_0/attention/self/query/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      @   
ĺ
1bert/encoder/layer_0/attention/self/query/ReshapeReshape@bert/encoder/layer_0/attention/self/query/Reshape/ReadVariableOp7bert/encoder/layer_0/attention/self/query/Reshape/shape*
T0*#
_output_shapes
:@
Ň
@bert/encoder/layer_0/attention/self/query/bias/Initializer/zerosConst*A
_class7
53loc:@bert/encoder/layer_0/attention/self/query/bias*
_output_shapes	
:*
dtype0*
valueB*    
ř
.bert/encoder/layer_0/attention/self/query/biasVarHandleOp*A
_class7
53loc:@bert/encoder/layer_0/attention/self/query/bias*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.bert/encoder/layer_0/attention/self/query/bias
­
Obert/encoder/layer_0/attention/self/query/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp.bert/encoder/layer_0/attention/self/query/bias*
_output_shapes
: 
Č
5bert/encoder/layer_0/attention/self/query/bias/AssignAssignVariableOp.bert/encoder/layer_0/attention/self/query/bias@bert/encoder/layer_0/attention/self/query/bias/Initializer/zeros*
dtype0
Ž
Bbert/encoder/layer_0/attention/self/query/bias/Read/ReadVariableOpReadVariableOp.bert/encoder/layer_0/attention/self/query/bias*
_output_shapes	
:*
dtype0
Ž
Bbert/encoder/layer_0/attention/self/query/Reshape_1/ReadVariableOpReadVariableOp.bert/encoder/layer_0/attention/self/query/bias*
_output_shapes	
:*
dtype0

9bert/encoder/layer_0/attention/self/query/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   
ć
3bert/encoder/layer_0/attention/self/query/Reshape_1ReshapeBbert/encoder/layer_0/attention/self/query/Reshape_1/ReadVariableOp9bert/encoder/layer_0/attention/self/query/Reshape_1/shape*
T0*
_output_shapes

:@

7bert/encoder/layer_0/attention/self/query/einsum/EinsumEinsum)bert/embeddings/LayerNorm/batchnorm/add_11bert/encoder/layer_0/attention/self/query/Reshape*
N*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
equationabc,cde->abde
ß
-bert/encoder/layer_0/attention/self/query/addAddV27bert/encoder/layer_0/attention/self/query/einsum/Einsum3bert/encoder/layer_0/attention/self/query/Reshape_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@

+bert/encoder/layer_0/attention/self/Shape_3Shape)bert/embeddings/LayerNorm/batchnorm/add_1*
T0*
_output_shapes
:

9bert/encoder/layer_0/attention/self/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 

;bert/encoder/layer_0/attention/self/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:

;bert/encoder/layer_0/attention/self/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
ë
3bert/encoder/layer_0/attention/self/strided_slice_3StridedSlice+bert/encoder/layer_0/attention/self/Shape_39bert/encoder/layer_0/attention/self/strided_slice_3/stack;bert/encoder/layer_0/attention/self/strided_slice_3/stack_1;bert/encoder/layer_0/attention/self/strided_slice_3/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
ĺ
Qbert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal/shapeConst*A
_class7
53loc:@bert/encoder/layer_0/attention/self/key/kernel*
_output_shapes
:*
dtype0*
valueB"      
Ř
Pbert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal/meanConst*A
_class7
53loc:@bert/encoder/layer_0/attention/self/key/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
Ú
Rbert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal/stddevConst*A
_class7
53loc:@bert/encoder/layer_0/attention/self/key/kernel*
_output_shapes
: *
dtype0*
valueB
 *
×Ł<
×
[bert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal/shape*
T0*A
_class7
53loc:@bert/encoder/layer_0/attention/self/key/kernel* 
_output_shapes
:
*
dtype0*
seed˛*
seed2
ő
Obert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal/stddev*
T0*A
_class7
53loc:@bert/encoder/layer_0/attention/self/key/kernel* 
_output_shapes
:

ă
Kbert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normalAddObert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal/mean*
T0*A
_class7
53loc:@bert/encoder/layer_0/attention/self/key/kernel* 
_output_shapes
:

ý
.bert/encoder/layer_0/attention/self/key/kernelVarHandleOp*A
_class7
53loc:@bert/encoder/layer_0/attention/self/key/kernel*
_output_shapes
: *
dtype0*
shape:
*?
shared_name0.bert/encoder/layer_0/attention/self/key/kernel
­
Obert/encoder/layer_0/attention/self/key/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp.bert/encoder/layer_0/attention/self/key/kernel*
_output_shapes
: 
Ó
5bert/encoder/layer_0/attention/self/key/kernel/AssignAssignVariableOp.bert/encoder/layer_0/attention/self/key/kernelKbert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal*
dtype0
ł
Bbert/encoder/layer_0/attention/self/key/kernel/Read/ReadVariableOpReadVariableOp.bert/encoder/layer_0/attention/self/key/kernel* 
_output_shapes
:
*
dtype0
Ż
>bert/encoder/layer_0/attention/self/key/Reshape/ReadVariableOpReadVariableOp.bert/encoder/layer_0/attention/self/key/kernel* 
_output_shapes
:
*
dtype0

5bert/encoder/layer_0/attention/self/key/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      @   
ß
/bert/encoder/layer_0/attention/self/key/ReshapeReshape>bert/encoder/layer_0/attention/self/key/Reshape/ReadVariableOp5bert/encoder/layer_0/attention/self/key/Reshape/shape*
T0*#
_output_shapes
:@
Î
>bert/encoder/layer_0/attention/self/key/bias/Initializer/zerosConst*?
_class5
31loc:@bert/encoder/layer_0/attention/self/key/bias*
_output_shapes	
:*
dtype0*
valueB*    
ň
,bert/encoder/layer_0/attention/self/key/biasVarHandleOp*?
_class5
31loc:@bert/encoder/layer_0/attention/self/key/bias*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,bert/encoder/layer_0/attention/self/key/bias
Š
Mbert/encoder/layer_0/attention/self/key/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp,bert/encoder/layer_0/attention/self/key/bias*
_output_shapes
: 
Â
3bert/encoder/layer_0/attention/self/key/bias/AssignAssignVariableOp,bert/encoder/layer_0/attention/self/key/bias>bert/encoder/layer_0/attention/self/key/bias/Initializer/zeros*
dtype0
Ş
@bert/encoder/layer_0/attention/self/key/bias/Read/ReadVariableOpReadVariableOp,bert/encoder/layer_0/attention/self/key/bias*
_output_shapes	
:*
dtype0
Ş
@bert/encoder/layer_0/attention/self/key/Reshape_1/ReadVariableOpReadVariableOp,bert/encoder/layer_0/attention/self/key/bias*
_output_shapes	
:*
dtype0

7bert/encoder/layer_0/attention/self/key/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   
ŕ
1bert/encoder/layer_0/attention/self/key/Reshape_1Reshape@bert/encoder/layer_0/attention/self/key/Reshape_1/ReadVariableOp7bert/encoder/layer_0/attention/self/key/Reshape_1/shape*
T0*
_output_shapes

:@
ü
5bert/encoder/layer_0/attention/self/key/einsum/EinsumEinsum)bert/embeddings/LayerNorm/batchnorm/add_1/bert/encoder/layer_0/attention/self/key/Reshape*
N*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
equationabc,cde->abde
Ů
+bert/encoder/layer_0/attention/self/key/addAddV25bert/encoder/layer_0/attention/self/key/einsum/Einsum1bert/encoder/layer_0/attention/self/key/Reshape_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@

+bert/encoder/layer_0/attention/self/Shape_4Shape)bert/embeddings/LayerNorm/batchnorm/add_1*
T0*
_output_shapes
:

9bert/encoder/layer_0/attention/self/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 

;bert/encoder/layer_0/attention/self/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:

;bert/encoder/layer_0/attention/self/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
ë
3bert/encoder/layer_0/attention/self/strided_slice_4StridedSlice+bert/encoder/layer_0/attention/self/Shape_49bert/encoder/layer_0/attention/self/strided_slice_4/stack;bert/encoder/layer_0/attention/self/strided_slice_4/stack_1;bert/encoder/layer_0/attention/self/strided_slice_4/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
é
Sbert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal/shapeConst*C
_class9
75loc:@bert/encoder/layer_0/attention/self/value/kernel*
_output_shapes
:*
dtype0*
valueB"      
Ü
Rbert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal/meanConst*C
_class9
75loc:@bert/encoder/layer_0/attention/self/value/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
Ţ
Tbert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal/stddevConst*C
_class9
75loc:@bert/encoder/layer_0/attention/self/value/kernel*
_output_shapes
: *
dtype0*
valueB
 *
×Ł<
Ý
]bert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal/shape*
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/self/value/kernel* 
_output_shapes
:
*
dtype0*
seed˛*
seed2
ý
Qbert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/self/value/kernel* 
_output_shapes
:

ë
Mbert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normalAddQbert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@bert/encoder/layer_0/attention/self/value/kernel* 
_output_shapes
:


0bert/encoder/layer_0/attention/self/value/kernelVarHandleOp*C
_class9
75loc:@bert/encoder/layer_0/attention/self/value/kernel*
_output_shapes
: *
dtype0*
shape:
*A
shared_name20bert/encoder/layer_0/attention/self/value/kernel
ą
Qbert/encoder/layer_0/attention/self/value/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp0bert/encoder/layer_0/attention/self/value/kernel*
_output_shapes
: 
Ů
7bert/encoder/layer_0/attention/self/value/kernel/AssignAssignVariableOp0bert/encoder/layer_0/attention/self/value/kernelMbert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal*
dtype0
ˇ
Dbert/encoder/layer_0/attention/self/value/kernel/Read/ReadVariableOpReadVariableOp0bert/encoder/layer_0/attention/self/value/kernel* 
_output_shapes
:
*
dtype0
ł
@bert/encoder/layer_0/attention/self/value/Reshape/ReadVariableOpReadVariableOp0bert/encoder/layer_0/attention/self/value/kernel* 
_output_shapes
:
*
dtype0

7bert/encoder/layer_0/attention/self/value/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      @   
ĺ
1bert/encoder/layer_0/attention/self/value/ReshapeReshape@bert/encoder/layer_0/attention/self/value/Reshape/ReadVariableOp7bert/encoder/layer_0/attention/self/value/Reshape/shape*
T0*#
_output_shapes
:@
Ň
@bert/encoder/layer_0/attention/self/value/bias/Initializer/zerosConst*A
_class7
53loc:@bert/encoder/layer_0/attention/self/value/bias*
_output_shapes	
:*
dtype0*
valueB*    
ř
.bert/encoder/layer_0/attention/self/value/biasVarHandleOp*A
_class7
53loc:@bert/encoder/layer_0/attention/self/value/bias*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.bert/encoder/layer_0/attention/self/value/bias
­
Obert/encoder/layer_0/attention/self/value/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp.bert/encoder/layer_0/attention/self/value/bias*
_output_shapes
: 
Č
5bert/encoder/layer_0/attention/self/value/bias/AssignAssignVariableOp.bert/encoder/layer_0/attention/self/value/bias@bert/encoder/layer_0/attention/self/value/bias/Initializer/zeros*
dtype0
Ž
Bbert/encoder/layer_0/attention/self/value/bias/Read/ReadVariableOpReadVariableOp.bert/encoder/layer_0/attention/self/value/bias*
_output_shapes	
:*
dtype0
Ž
Bbert/encoder/layer_0/attention/self/value/Reshape_1/ReadVariableOpReadVariableOp.bert/encoder/layer_0/attention/self/value/bias*
_output_shapes	
:*
dtype0

9bert/encoder/layer_0/attention/self/value/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   
ć
3bert/encoder/layer_0/attention/self/value/Reshape_1ReshapeBbert/encoder/layer_0/attention/self/value/Reshape_1/ReadVariableOp9bert/encoder/layer_0/attention/self/value/Reshape_1/shape*
T0*
_output_shapes

:@

7bert/encoder/layer_0/attention/self/value/einsum/EinsumEinsum)bert/embeddings/LayerNorm/batchnorm/add_11bert/encoder/layer_0/attention/self/value/Reshape*
N*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
equationabc,cde->abde
ß
-bert/encoder/layer_0/attention/self/value/addAddV27bert/encoder/layer_0/attention/self/value/einsum/Einsum3bert/encoder/layer_0/attention/self/value/Reshape_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@
ű
1bert/encoder/layer_0/attention/self/einsum/EinsumEinsum+bert/encoder/layer_0/attention/self/key/add-bert/encoder/layer_0/attention/self/query/add*
N*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
equationBTNH,BFNH->BNFT
n
)bert/encoder/layer_0/attention/self/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >
Č
'bert/encoder/layer_0/attention/self/MulMul1bert/encoder/layer_0/attention/self/einsum/Einsum)bert/encoder/layer_0/attention/self/Mul/y*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
|
2bert/encoder/layer_0/attention/self/ExpandDims/dimConst*
_output_shapes
:*
dtype0*
valueB:
ž
.bert/encoder/layer_0/attention/self/ExpandDims
ExpandDimsbert/encoder/mul2bert/encoder/layer_0/attention/self/ExpandDims/dim*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
n
)bert/encoder/layer_0/attention/self/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
Ĺ
'bert/encoder/layer_0/attention/self/subSub)bert/encoder/layer_0/attention/self/sub/x.bert/encoder/layer_0/attention/self/ExpandDims*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
+bert/encoder/layer_0/attention/self/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 * @Ć
Â
)bert/encoder/layer_0/attention/self/mul_1Mul'bert/encoder/layer_0/attention/self/sub+bert/encoder/layer_0/attention/self/mul_1/y*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ŕ
'bert/encoder/layer_0/attention/self/addAddV2'bert/encoder/layer_0/attention/self/Mul)bert/encoder/layer_0/attention/self/mul_1*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙

+bert/encoder/layer_0/attention/self/SoftmaxSoftmax'bert/encoder/layer_0/attention/self/add*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
ü
3bert/encoder/layer_0/attention/self/einsum_1/EinsumEinsum+bert/encoder/layer_0/attention/self/Softmax-bert/encoder/layer_0/attention/self/value/add*
N*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
equationBNFT,BTNH->BFNH
í
Ubert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal/shapeConst*E
_class;
97loc:@bert/encoder/layer_0/attention/output/dense/kernel*
_output_shapes
:*
dtype0*
valueB"      
ŕ
Tbert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal/meanConst*E
_class;
97loc:@bert/encoder/layer_0/attention/output/dense/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
â
Vbert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal/stddevConst*E
_class;
97loc:@bert/encoder/layer_0/attention/output/dense/kernel*
_output_shapes
: *
dtype0*
valueB
 *
×Ł<
ă
_bert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalUbert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal/shape*
T0*E
_class;
97loc:@bert/encoder/layer_0/attention/output/dense/kernel* 
_output_shapes
:
*
dtype0*
seed˛*
seed2

Sbert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal/mulMul_bert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalVbert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*E
_class;
97loc:@bert/encoder/layer_0/attention/output/dense/kernel* 
_output_shapes
:

ó
Obert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normalAddSbert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal/mulTbert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal/mean*
T0*E
_class;
97loc:@bert/encoder/layer_0/attention/output/dense/kernel* 
_output_shapes
:


2bert/encoder/layer_0/attention/output/dense/kernelVarHandleOp*E
_class;
97loc:@bert/encoder/layer_0/attention/output/dense/kernel*
_output_shapes
: *
dtype0*
shape:
*C
shared_name42bert/encoder/layer_0/attention/output/dense/kernel
ľ
Sbert/encoder/layer_0/attention/output/dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp2bert/encoder/layer_0/attention/output/dense/kernel*
_output_shapes
: 
ß
9bert/encoder/layer_0/attention/output/dense/kernel/AssignAssignVariableOp2bert/encoder/layer_0/attention/output/dense/kernelObert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal*
dtype0
ť
Fbert/encoder/layer_0/attention/output/dense/kernel/Read/ReadVariableOpReadVariableOp2bert/encoder/layer_0/attention/output/dense/kernel* 
_output_shapes
:
*
dtype0
ˇ
Bbert/encoder/layer_0/attention/output/dense/Reshape/ReadVariableOpReadVariableOp2bert/encoder/layer_0/attention/output/dense/kernel* 
_output_shapes
:
*
dtype0

9bert/encoder/layer_0/attention/output/dense/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   @      
ë
3bert/encoder/layer_0/attention/output/dense/ReshapeReshapeBbert/encoder/layer_0/attention/output/dense/Reshape/ReadVariableOp9bert/encoder/layer_0/attention/output/dense/Reshape/shape*
T0*#
_output_shapes
:@
Ö
Bbert/encoder/layer_0/attention/output/dense/bias/Initializer/zerosConst*C
_class9
75loc:@bert/encoder/layer_0/attention/output/dense/bias*
_output_shapes	
:*
dtype0*
valueB*    
ţ
0bert/encoder/layer_0/attention/output/dense/biasVarHandleOp*C
_class9
75loc:@bert/encoder/layer_0/attention/output/dense/bias*
_output_shapes
: *
dtype0*
shape:*A
shared_name20bert/encoder/layer_0/attention/output/dense/bias
ą
Qbert/encoder/layer_0/attention/output/dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp0bert/encoder/layer_0/attention/output/dense/bias*
_output_shapes
: 
Î
7bert/encoder/layer_0/attention/output/dense/bias/AssignAssignVariableOp0bert/encoder/layer_0/attention/output/dense/biasBbert/encoder/layer_0/attention/output/dense/bias/Initializer/zeros*
dtype0
˛
Dbert/encoder/layer_0/attention/output/dense/bias/Read/ReadVariableOpReadVariableOp0bert/encoder/layer_0/attention/output/dense/bias*
_output_shapes	
:*
dtype0

3bert/encoder/layer_0/attention/output/einsum/EinsumEinsum3bert/encoder/layer_0/attention/self/einsum_1/Einsum3bert/encoder/layer_0/attention/output/dense/Reshape*
N*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙*
equationBFNH,NHD->BFD
Ś
8bert/encoder/layer_0/attention/output/add/ReadVariableOpReadVariableOp0bert/encoder/layer_0/attention/output/dense/bias*
_output_shapes	
:*
dtype0
Ů
)bert/encoder/layer_0/attention/output/addAddV23bert/encoder/layer_0/attention/output/einsum/Einsum8bert/encoder/layer_0/attention/output/add/ReadVariableOp*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
Â
+bert/encoder/layer_0/attention/output/add_1AddV2)bert/encoder/layer_0/attention/output/add)bert/embeddings/LayerNorm/batchnorm/add_1*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ţ
Fbert/encoder/layer_0/attention/output/LayerNorm/beta/Initializer/zerosConst*G
_class=
;9loc:@bert/encoder/layer_0/attention/output/LayerNorm/beta*
_output_shapes	
:*
dtype0*
valueB*    

4bert/encoder/layer_0/attention/output/LayerNorm/betaVarHandleOp*G
_class=
;9loc:@bert/encoder/layer_0/attention/output/LayerNorm/beta*
_output_shapes
: *
dtype0*
shape:*E
shared_name64bert/encoder/layer_0/attention/output/LayerNorm/beta
š
Ubert/encoder/layer_0/attention/output/LayerNorm/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp4bert/encoder/layer_0/attention/output/LayerNorm/beta*
_output_shapes
: 
Ú
;bert/encoder/layer_0/attention/output/LayerNorm/beta/AssignAssignVariableOp4bert/encoder/layer_0/attention/output/LayerNorm/betaFbert/encoder/layer_0/attention/output/LayerNorm/beta/Initializer/zeros*
dtype0
ş
Hbert/encoder/layer_0/attention/output/LayerNorm/beta/Read/ReadVariableOpReadVariableOp4bert/encoder/layer_0/attention/output/LayerNorm/beta*
_output_shapes	
:*
dtype0
ß
Fbert/encoder/layer_0/attention/output/LayerNorm/gamma/Initializer/onesConst*H
_class>
<:loc:@bert/encoder/layer_0/attention/output/LayerNorm/gamma*
_output_shapes	
:*
dtype0*
valueB*  ?

5bert/encoder/layer_0/attention/output/LayerNorm/gammaVarHandleOp*H
_class>
<:loc:@bert/encoder/layer_0/attention/output/LayerNorm/gamma*
_output_shapes
: *
dtype0*
shape:*F
shared_name75bert/encoder/layer_0/attention/output/LayerNorm/gamma
ť
Vbert/encoder/layer_0/attention/output/LayerNorm/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp5bert/encoder/layer_0/attention/output/LayerNorm/gamma*
_output_shapes
: 
Ü
<bert/encoder/layer_0/attention/output/LayerNorm/gamma/AssignAssignVariableOp5bert/encoder/layer_0/attention/output/LayerNorm/gammaFbert/encoder/layer_0/attention/output/LayerNorm/gamma/Initializer/ones*
dtype0
ź
Ibert/encoder/layer_0/attention/output/LayerNorm/gamma/Read/ReadVariableOpReadVariableOp5bert/encoder/layer_0/attention/output/LayerNorm/gamma*
_output_shapes	
:*
dtype0

Nbert/encoder/layer_0/attention/output/LayerNorm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:

<bert/encoder/layer_0/attention/output/LayerNorm/moments/meanMean+bert/encoder/layer_0/attention/output/add_1Nbert/encoder/layer_0/attention/output/LayerNorm/moments/mean/reduction_indices*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(
É
Dbert/encoder/layer_0/attention/output/LayerNorm/moments/StopGradientStopGradient<bert/encoder/layer_0/attention/output/LayerNorm/moments/mean*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ibert/encoder/layer_0/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference+bert/encoder/layer_0/attention/output/add_1Dbert/encoder/layer_0/attention/output/LayerNorm/moments/StopGradient*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙

Rbert/encoder/layer_0/attention/output/LayerNorm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
Ż
@bert/encoder/layer_0/attention/output/LayerNorm/moments/varianceMeanIbert/encoder/layer_0/attention/output/LayerNorm/moments/SquaredDifferenceRbert/encoder/layer_0/attention/output/LayerNorm/moments/variance/reduction_indices*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(

?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ěź+

=bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/addAddV2@bert/encoder/layer_0/attention/output/LayerNorm/moments/variance?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add/y*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/RsqrtRsqrt=bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
ż
Lbert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mul/ReadVariableOpReadVariableOp5bert/encoder/layer_0/attention/output/LayerNorm/gamma*
_output_shapes	
:*
dtype0

=bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mulMul?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/RsqrtLbert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mul/ReadVariableOp*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
ę
?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mul_1Mul+bert/encoder/layer_0/attention/output/add_1=bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mul*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
ű
?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mul_2Mul<bert/encoder/layer_0/attention/output/LayerNorm/moments/mean=bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mul*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
ş
Hbert/encoder/layer_0/attention/output/LayerNorm/batchnorm/ReadVariableOpReadVariableOp4bert/encoder/layer_0/attention/output/LayerNorm/beta*
_output_shapes	
:*
dtype0

=bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/subSubHbert/encoder/layer_0/attention/output/LayerNorm/batchnorm/ReadVariableOp?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mul_2*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙

?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add_1AddV2?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/mul_1=bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/sub*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙

'bert/encoder/layer_0/intermediate/ShapeShape?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add_1*
T0*
_output_shapes
:

5bert/encoder/layer_0/intermediate/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 

7bert/encoder/layer_0/intermediate/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:

7bert/encoder/layer_0/intermediate/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
×
/bert/encoder/layer_0/intermediate/strided_sliceStridedSlice'bert/encoder/layer_0/intermediate/Shape5bert/encoder/layer_0/intermediate/strided_slice/stack7bert/encoder/layer_0/intermediate/strided_slice/stack_17bert/encoder/layer_0/intermediate/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
ĺ
Qbert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal/shapeConst*A
_class7
53loc:@bert/encoder/layer_0/intermediate/dense/kernel*
_output_shapes
:*
dtype0*
valueB"      
Ř
Pbert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal/meanConst*A
_class7
53loc:@bert/encoder/layer_0/intermediate/dense/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
Ú
Rbert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal/stddevConst*A
_class7
53loc:@bert/encoder/layer_0/intermediate/dense/kernel*
_output_shapes
: *
dtype0*
valueB
 *
×Ł<
×
[bert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal/shape*
T0*A
_class7
53loc:@bert/encoder/layer_0/intermediate/dense/kernel* 
_output_shapes
:
*
dtype0*
seed˛*
seed2
ő
Obert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal/stddev*
T0*A
_class7
53loc:@bert/encoder/layer_0/intermediate/dense/kernel* 
_output_shapes
:

ă
Kbert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normalAddObert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal/mean*
T0*A
_class7
53loc:@bert/encoder/layer_0/intermediate/dense/kernel* 
_output_shapes
:

ý
.bert/encoder/layer_0/intermediate/dense/kernelVarHandleOp*A
_class7
53loc:@bert/encoder/layer_0/intermediate/dense/kernel*
_output_shapes
: *
dtype0*
shape:
*?
shared_name0.bert/encoder/layer_0/intermediate/dense/kernel
­
Obert/encoder/layer_0/intermediate/dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp.bert/encoder/layer_0/intermediate/dense/kernel*
_output_shapes
: 
Ó
5bert/encoder/layer_0/intermediate/dense/kernel/AssignAssignVariableOp.bert/encoder/layer_0/intermediate/dense/kernelKbert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal*
dtype0
ł
Bbert/encoder/layer_0/intermediate/dense/kernel/Read/ReadVariableOpReadVariableOp.bert/encoder/layer_0/intermediate/dense/kernel* 
_output_shapes
:
*
dtype0
Î
>bert/encoder/layer_0/intermediate/dense/bias/Initializer/zerosConst*?
_class5
31loc:@bert/encoder/layer_0/intermediate/dense/bias*
_output_shapes	
:*
dtype0*
valueB*    
ň
,bert/encoder/layer_0/intermediate/dense/biasVarHandleOp*?
_class5
31loc:@bert/encoder/layer_0/intermediate/dense/bias*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,bert/encoder/layer_0/intermediate/dense/bias
Š
Mbert/encoder/layer_0/intermediate/dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp,bert/encoder/layer_0/intermediate/dense/bias*
_output_shapes
: 
Â
3bert/encoder/layer_0/intermediate/dense/bias/AssignAssignVariableOp,bert/encoder/layer_0/intermediate/dense/bias>bert/encoder/layer_0/intermediate/dense/bias/Initializer/zeros*
dtype0
Ş
@bert/encoder/layer_0/intermediate/dense/bias/Read/ReadVariableOpReadVariableOp,bert/encoder/layer_0/intermediate/dense/bias*
_output_shapes	
:*
dtype0
Ż
>bert/encoder/layer_0/intermediate/einsum/Einsum/ReadVariableOpReadVariableOp.bert/encoder/layer_0/intermediate/dense/kernel* 
_output_shapes
:
*
dtype0

/bert/encoder/layer_0/intermediate/einsum/EinsumEinsum?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add_1>bert/encoder/layer_0/intermediate/einsum/Einsum/ReadVariableOp*
N*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙*
equationabc,cd->abd

4bert/encoder/layer_0/intermediate/add/ReadVariableOpReadVariableOp,bert/encoder/layer_0/intermediate/dense/bias*
_output_shapes	
:*
dtype0
Í
%bert/encoder/layer_0/intermediate/addAddV2/bert/encoder/layer_0/intermediate/einsum/Einsum4bert/encoder/layer_0/intermediate/add/ReadVariableOp*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
l
'bert/encoder/layer_0/intermediate/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@
´
%bert/encoder/layer_0/intermediate/PowPow%bert/encoder/layer_0/intermediate/add'bert/encoder/layer_0/intermediate/Pow/y*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
l
'bert/encoder/layer_0/intermediate/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=
´
%bert/encoder/layer_0/intermediate/mulMul'bert/encoder/layer_0/intermediate/mul/x%bert/encoder/layer_0/intermediate/Pow*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
ś
'bert/encoder/layer_0/intermediate/add_1AddV2%bert/encoder/layer_0/intermediate/add%bert/encoder/layer_0/intermediate/mul*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
n
)bert/encoder/layer_0/intermediate/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 **BL?
ş
'bert/encoder/layer_0/intermediate/mul_1Mul)bert/encoder/layer_0/intermediate/mul_1/x'bert/encoder/layer_0/intermediate/add_1*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙

&bert/encoder/layer_0/intermediate/TanhTanh'bert/encoder/layer_0/intermediate/mul_1*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
n
)bert/encoder/layer_0/intermediate/add_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
ť
'bert/encoder/layer_0/intermediate/add_2AddV2)bert/encoder/layer_0/intermediate/add_2/x&bert/encoder/layer_0/intermediate/Tanh*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
n
)bert/encoder/layer_0/intermediate/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
ş
'bert/encoder/layer_0/intermediate/mul_2Mul)bert/encoder/layer_0/intermediate/mul_2/x'bert/encoder/layer_0/intermediate/add_2*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
ś
'bert/encoder/layer_0/intermediate/mul_3Mul%bert/encoder/layer_0/intermediate/add'bert/encoder/layer_0/intermediate/mul_2*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
x
!bert/encoder/layer_0/output/ShapeShape'bert/encoder/layer_0/intermediate/mul_3*
T0*
_output_shapes
:
y
/bert/encoder/layer_0/output/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
{
1bert/encoder/layer_0/output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
{
1bert/encoder/layer_0/output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
š
)bert/encoder/layer_0/output/strided_sliceStridedSlice!bert/encoder/layer_0/output/Shape/bert/encoder/layer_0/output/strided_slice/stack1bert/encoder/layer_0/output/strided_slice/stack_11bert/encoder/layer_0/output/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Ů
Kbert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal/shapeConst*;
_class1
/-loc:@bert/encoder/layer_0/output/dense/kernel*
_output_shapes
:*
dtype0*
valueB"      
Ě
Jbert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal/meanConst*;
_class1
/-loc:@bert/encoder/layer_0/output/dense/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
Î
Lbert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal/stddevConst*;
_class1
/-loc:@bert/encoder/layer_0/output/dense/kernel*
_output_shapes
: *
dtype0*
valueB
 *
×Ł<
Ĺ
Ubert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalKbert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal/shape*
T0*;
_class1
/-loc:@bert/encoder/layer_0/output/dense/kernel* 
_output_shapes
:
*
dtype0*
seed˛*
seed2
Ý
Ibert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal/mulMulUbert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalLbert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*;
_class1
/-loc:@bert/encoder/layer_0/output/dense/kernel* 
_output_shapes
:

Ë
Ebert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normalAddIbert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal/mulJbert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal/mean*
T0*;
_class1
/-loc:@bert/encoder/layer_0/output/dense/kernel* 
_output_shapes
:

ë
(bert/encoder/layer_0/output/dense/kernelVarHandleOp*;
_class1
/-loc:@bert/encoder/layer_0/output/dense/kernel*
_output_shapes
: *
dtype0*
shape:
*9
shared_name*(bert/encoder/layer_0/output/dense/kernel
Ą
Ibert/encoder/layer_0/output/dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp(bert/encoder/layer_0/output/dense/kernel*
_output_shapes
: 
Á
/bert/encoder/layer_0/output/dense/kernel/AssignAssignVariableOp(bert/encoder/layer_0/output/dense/kernelEbert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal*
dtype0
§
<bert/encoder/layer_0/output/dense/kernel/Read/ReadVariableOpReadVariableOp(bert/encoder/layer_0/output/dense/kernel* 
_output_shapes
:
*
dtype0
Â
8bert/encoder/layer_0/output/dense/bias/Initializer/zerosConst*9
_class/
-+loc:@bert/encoder/layer_0/output/dense/bias*
_output_shapes	
:*
dtype0*
valueB*    
ŕ
&bert/encoder/layer_0/output/dense/biasVarHandleOp*9
_class/
-+loc:@bert/encoder/layer_0/output/dense/bias*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&bert/encoder/layer_0/output/dense/bias

Gbert/encoder/layer_0/output/dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp&bert/encoder/layer_0/output/dense/bias*
_output_shapes
: 
°
-bert/encoder/layer_0/output/dense/bias/AssignAssignVariableOp&bert/encoder/layer_0/output/dense/bias8bert/encoder/layer_0/output/dense/bias/Initializer/zeros*
dtype0

:bert/encoder/layer_0/output/dense/bias/Read/ReadVariableOpReadVariableOp&bert/encoder/layer_0/output/dense/bias*
_output_shapes	
:*
dtype0
Ł
8bert/encoder/layer_0/output/einsum/Einsum/ReadVariableOpReadVariableOp(bert/encoder/layer_0/output/dense/kernel* 
_output_shapes
:
*
dtype0
ň
)bert/encoder/layer_0/output/einsum/EinsumEinsum'bert/encoder/layer_0/intermediate/mul_38bert/encoder/layer_0/output/einsum/Einsum/ReadVariableOp*
N*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙*
equationabc,cd->abd

.bert/encoder/layer_0/output/add/ReadVariableOpReadVariableOp&bert/encoder/layer_0/output/dense/bias*
_output_shapes	
:*
dtype0
ť
bert/encoder/layer_0/output/addAddV2)bert/encoder/layer_0/output/einsum/Einsum.bert/encoder/layer_0/output/add/ReadVariableOp*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ä
!bert/encoder/layer_0/output/add_1AddV2bert/encoder/layer_0/output/add?bert/encoder/layer_0/attention/output/LayerNorm/batchnorm/add_1*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ę
<bert/encoder/layer_0/output/LayerNorm/beta/Initializer/zerosConst*=
_class3
1/loc:@bert/encoder/layer_0/output/LayerNorm/beta*
_output_shapes	
:*
dtype0*
valueB*    
ě
*bert/encoder/layer_0/output/LayerNorm/betaVarHandleOp*=
_class3
1/loc:@bert/encoder/layer_0/output/LayerNorm/beta*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*bert/encoder/layer_0/output/LayerNorm/beta
Ľ
Kbert/encoder/layer_0/output/LayerNorm/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp*bert/encoder/layer_0/output/LayerNorm/beta*
_output_shapes
: 
ź
1bert/encoder/layer_0/output/LayerNorm/beta/AssignAssignVariableOp*bert/encoder/layer_0/output/LayerNorm/beta<bert/encoder/layer_0/output/LayerNorm/beta/Initializer/zeros*
dtype0
Ś
>bert/encoder/layer_0/output/LayerNorm/beta/Read/ReadVariableOpReadVariableOp*bert/encoder/layer_0/output/LayerNorm/beta*
_output_shapes	
:*
dtype0
Ë
<bert/encoder/layer_0/output/LayerNorm/gamma/Initializer/onesConst*>
_class4
20loc:@bert/encoder/layer_0/output/LayerNorm/gamma*
_output_shapes	
:*
dtype0*
valueB*  ?
ď
+bert/encoder/layer_0/output/LayerNorm/gammaVarHandleOp*>
_class4
20loc:@bert/encoder/layer_0/output/LayerNorm/gamma*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+bert/encoder/layer_0/output/LayerNorm/gamma
§
Lbert/encoder/layer_0/output/LayerNorm/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp+bert/encoder/layer_0/output/LayerNorm/gamma*
_output_shapes
: 
ž
2bert/encoder/layer_0/output/LayerNorm/gamma/AssignAssignVariableOp+bert/encoder/layer_0/output/LayerNorm/gamma<bert/encoder/layer_0/output/LayerNorm/gamma/Initializer/ones*
dtype0
¨
?bert/encoder/layer_0/output/LayerNorm/gamma/Read/ReadVariableOpReadVariableOp+bert/encoder/layer_0/output/LayerNorm/gamma*
_output_shapes	
:*
dtype0

Dbert/encoder/layer_0/output/LayerNorm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
ë
2bert/encoder/layer_0/output/LayerNorm/moments/meanMean!bert/encoder/layer_0/output/add_1Dbert/encoder/layer_0/output/LayerNorm/moments/mean/reduction_indices*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(
ľ
:bert/encoder/layer_0/output/LayerNorm/moments/StopGradientStopGradient2bert/encoder/layer_0/output/LayerNorm/moments/mean*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
ë
?bert/encoder/layer_0/output/LayerNorm/moments/SquaredDifferenceSquaredDifference!bert/encoder/layer_0/output/add_1:bert/encoder/layer_0/output/LayerNorm/moments/StopGradient*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙

Hbert/encoder/layer_0/output/LayerNorm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:

6bert/encoder/layer_0/output/LayerNorm/moments/varianceMean?bert/encoder/layer_0/output/LayerNorm/moments/SquaredDifferenceHbert/encoder/layer_0/output/LayerNorm/moments/variance/reduction_indices*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(
z
5bert/encoder/layer_0/output/LayerNorm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ěź+
â
3bert/encoder/layer_0/output/LayerNorm/batchnorm/addAddV26bert/encoder/layer_0/output/LayerNorm/moments/variance5bert/encoder/layer_0/output/LayerNorm/batchnorm/add/y*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ş
5bert/encoder/layer_0/output/LayerNorm/batchnorm/RsqrtRsqrt3bert/encoder/layer_0/output/LayerNorm/batchnorm/add*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ť
Bbert/encoder/layer_0/output/LayerNorm/batchnorm/mul/ReadVariableOpReadVariableOp+bert/encoder/layer_0/output/LayerNorm/gamma*
_output_shapes	
:*
dtype0
í
3bert/encoder/layer_0/output/LayerNorm/batchnorm/mulMul5bert/encoder/layer_0/output/LayerNorm/batchnorm/RsqrtBbert/encoder/layer_0/output/LayerNorm/batchnorm/mul/ReadVariableOp*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ě
5bert/encoder/layer_0/output/LayerNorm/batchnorm/mul_1Mul!bert/encoder/layer_0/output/add_13bert/encoder/layer_0/output/LayerNorm/batchnorm/mul*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ý
5bert/encoder/layer_0/output/LayerNorm/batchnorm/mul_2Mul2bert/encoder/layer_0/output/LayerNorm/moments/mean3bert/encoder/layer_0/output/LayerNorm/batchnorm/mul*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
>bert/encoder/layer_0/output/LayerNorm/batchnorm/ReadVariableOpReadVariableOp*bert/encoder/layer_0/output/LayerNorm/beta*
_output_shapes	
:*
dtype0
é
3bert/encoder/layer_0/output/LayerNorm/batchnorm/subSub>bert/encoder/layer_0/output/LayerNorm/batchnorm/ReadVariableOp5bert/encoder/layer_0/output/LayerNorm/batchnorm/mul_2*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
â
5bert/encoder/layer_0/output/LayerNorm/batchnorm/add_1AddV25bert/encoder/layer_0/output/LayerNorm/batchnorm/mul_13bert/encoder/layer_0/output/LayerNorm/batchnorm/sub*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙

)bert/encoder/layer_1/attention/self/ShapeShape5bert/encoder/layer_0/output/LayerNorm/batchnorm/add_1*
T0*
_output_shapes
:

7bert/encoder/layer_1/attention/self/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 

9bert/encoder/layer_1/attention/self/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:

9bert/encoder/layer_1/attention/self/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
á
1bert/encoder/layer_1/attention/self/strided_sliceStridedSlice)bert/encoder/layer_1/attention/self/Shape7bert/encoder/layer_1/attention/self/strided_slice/stack9bert/encoder/layer_1/attention/self/strided_slice/stack_19bert/encoder/layer_1/attention/self/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

+bert/encoder/layer_1/attention/self/Shape_1Shape5bert/encoder/layer_0/output/LayerNorm/batchnorm/add_1*
T0*
_output_shapes
:

9bert/encoder/layer_1/attention/self/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 

;bert/encoder/layer_1/attention/self/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:

;bert/encoder/layer_1/attention/self/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
ë
3bert/encoder/layer_1/attention/self/strided_slice_1StridedSlice+bert/encoder/layer_1/attention/self/Shape_19bert/encoder/layer_1/attention/self/strided_slice_1/stack;bert/encoder/layer_1/attention/self/strided_slice_1/stack_1;bert/encoder/layer_1/attention/self/strided_slice_1/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask

+bert/encoder/layer_1/attention/self/Shape_2Shape5bert/encoder/layer_0/output/LayerNorm/batchnorm/add_1*
T0*
_output_shapes
:

9bert/encoder/layer_1/attention/self/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 

;bert/encoder/layer_1/attention/self/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:

;bert/encoder/layer_1/attention/self/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
ë
3bert/encoder/layer_1/attention/self/strided_slice_2StridedSlice+bert/encoder/layer_1/attention/self/Shape_29bert/encoder/layer_1/attention/self/strided_slice_2/stack;bert/encoder/layer_1/attention/self/strided_slice_2/stack_1;bert/encoder/layer_1/attention/self/strided_slice_2/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
é
Sbert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal/shapeConst*C
_class9
75loc:@bert/encoder/layer_1/attention/self/query/kernel*
_output_shapes
:*
dtype0*
valueB"      
Ü
Rbert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal/meanConst*C
_class9
75loc:@bert/encoder/layer_1/attention/self/query/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
Ţ
Tbert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal/stddevConst*C
_class9
75loc:@bert/encoder/layer_1/attention/self/query/kernel*
_output_shapes
: *
dtype0*
valueB
 *
×Ł<
Ý
]bert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal/shape*
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/self/query/kernel* 
_output_shapes
:
*
dtype0*
seed˛*
seed2	
ý
Qbert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/self/query/kernel* 
_output_shapes
:

ë
Mbert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normalAddQbert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/self/query/kernel* 
_output_shapes
:


0bert/encoder/layer_1/attention/self/query/kernelVarHandleOp*C
_class9
75loc:@bert/encoder/layer_1/attention/self/query/kernel*
_output_shapes
: *
dtype0*
shape:
*A
shared_name20bert/encoder/layer_1/attention/self/query/kernel
ą
Qbert/encoder/layer_1/attention/self/query/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp0bert/encoder/layer_1/attention/self/query/kernel*
_output_shapes
: 
Ů
7bert/encoder/layer_1/attention/self/query/kernel/AssignAssignVariableOp0bert/encoder/layer_1/attention/self/query/kernelMbert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal*
dtype0
ˇ
Dbert/encoder/layer_1/attention/self/query/kernel/Read/ReadVariableOpReadVariableOp0bert/encoder/layer_1/attention/self/query/kernel* 
_output_shapes
:
*
dtype0
ł
@bert/encoder/layer_1/attention/self/query/Reshape/ReadVariableOpReadVariableOp0bert/encoder/layer_1/attention/self/query/kernel* 
_output_shapes
:
*
dtype0

7bert/encoder/layer_1/attention/self/query/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      @   
ĺ
1bert/encoder/layer_1/attention/self/query/ReshapeReshape@bert/encoder/layer_1/attention/self/query/Reshape/ReadVariableOp7bert/encoder/layer_1/attention/self/query/Reshape/shape*
T0*#
_output_shapes
:@
Ň
@bert/encoder/layer_1/attention/self/query/bias/Initializer/zerosConst*A
_class7
53loc:@bert/encoder/layer_1/attention/self/query/bias*
_output_shapes	
:*
dtype0*
valueB*    
ř
.bert/encoder/layer_1/attention/self/query/biasVarHandleOp*A
_class7
53loc:@bert/encoder/layer_1/attention/self/query/bias*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.bert/encoder/layer_1/attention/self/query/bias
­
Obert/encoder/layer_1/attention/self/query/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp.bert/encoder/layer_1/attention/self/query/bias*
_output_shapes
: 
Č
5bert/encoder/layer_1/attention/self/query/bias/AssignAssignVariableOp.bert/encoder/layer_1/attention/self/query/bias@bert/encoder/layer_1/attention/self/query/bias/Initializer/zeros*
dtype0
Ž
Bbert/encoder/layer_1/attention/self/query/bias/Read/ReadVariableOpReadVariableOp.bert/encoder/layer_1/attention/self/query/bias*
_output_shapes	
:*
dtype0
Ž
Bbert/encoder/layer_1/attention/self/query/Reshape_1/ReadVariableOpReadVariableOp.bert/encoder/layer_1/attention/self/query/bias*
_output_shapes	
:*
dtype0

9bert/encoder/layer_1/attention/self/query/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   
ć
3bert/encoder/layer_1/attention/self/query/Reshape_1ReshapeBbert/encoder/layer_1/attention/self/query/Reshape_1/ReadVariableOp9bert/encoder/layer_1/attention/self/query/Reshape_1/shape*
T0*
_output_shapes

:@

7bert/encoder/layer_1/attention/self/query/einsum/EinsumEinsum5bert/encoder/layer_0/output/LayerNorm/batchnorm/add_11bert/encoder/layer_1/attention/self/query/Reshape*
N*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
equationabc,cde->abde
ß
-bert/encoder/layer_1/attention/self/query/addAddV27bert/encoder/layer_1/attention/self/query/einsum/Einsum3bert/encoder/layer_1/attention/self/query/Reshape_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@

+bert/encoder/layer_1/attention/self/Shape_3Shape5bert/encoder/layer_0/output/LayerNorm/batchnorm/add_1*
T0*
_output_shapes
:

9bert/encoder/layer_1/attention/self/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 

;bert/encoder/layer_1/attention/self/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:

;bert/encoder/layer_1/attention/self/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
ë
3bert/encoder/layer_1/attention/self/strided_slice_3StridedSlice+bert/encoder/layer_1/attention/self/Shape_39bert/encoder/layer_1/attention/self/strided_slice_3/stack;bert/encoder/layer_1/attention/self/strided_slice_3/stack_1;bert/encoder/layer_1/attention/self/strided_slice_3/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
ĺ
Qbert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal/shapeConst*A
_class7
53loc:@bert/encoder/layer_1/attention/self/key/kernel*
_output_shapes
:*
dtype0*
valueB"      
Ř
Pbert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal/meanConst*A
_class7
53loc:@bert/encoder/layer_1/attention/self/key/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
Ú
Rbert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal/stddevConst*A
_class7
53loc:@bert/encoder/layer_1/attention/self/key/kernel*
_output_shapes
: *
dtype0*
valueB
 *
×Ł<
×
[bert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal/shape*
T0*A
_class7
53loc:@bert/encoder/layer_1/attention/self/key/kernel* 
_output_shapes
:
*
dtype0*
seed˛*
seed2

ő
Obert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal/stddev*
T0*A
_class7
53loc:@bert/encoder/layer_1/attention/self/key/kernel* 
_output_shapes
:

ă
Kbert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normalAddObert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal/mean*
T0*A
_class7
53loc:@bert/encoder/layer_1/attention/self/key/kernel* 
_output_shapes
:

ý
.bert/encoder/layer_1/attention/self/key/kernelVarHandleOp*A
_class7
53loc:@bert/encoder/layer_1/attention/self/key/kernel*
_output_shapes
: *
dtype0*
shape:
*?
shared_name0.bert/encoder/layer_1/attention/self/key/kernel
­
Obert/encoder/layer_1/attention/self/key/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp.bert/encoder/layer_1/attention/self/key/kernel*
_output_shapes
: 
Ó
5bert/encoder/layer_1/attention/self/key/kernel/AssignAssignVariableOp.bert/encoder/layer_1/attention/self/key/kernelKbert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal*
dtype0
ł
Bbert/encoder/layer_1/attention/self/key/kernel/Read/ReadVariableOpReadVariableOp.bert/encoder/layer_1/attention/self/key/kernel* 
_output_shapes
:
*
dtype0
Ż
>bert/encoder/layer_1/attention/self/key/Reshape/ReadVariableOpReadVariableOp.bert/encoder/layer_1/attention/self/key/kernel* 
_output_shapes
:
*
dtype0

5bert/encoder/layer_1/attention/self/key/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      @   
ß
/bert/encoder/layer_1/attention/self/key/ReshapeReshape>bert/encoder/layer_1/attention/self/key/Reshape/ReadVariableOp5bert/encoder/layer_1/attention/self/key/Reshape/shape*
T0*#
_output_shapes
:@
Î
>bert/encoder/layer_1/attention/self/key/bias/Initializer/zerosConst*?
_class5
31loc:@bert/encoder/layer_1/attention/self/key/bias*
_output_shapes	
:*
dtype0*
valueB*    
ň
,bert/encoder/layer_1/attention/self/key/biasVarHandleOp*?
_class5
31loc:@bert/encoder/layer_1/attention/self/key/bias*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,bert/encoder/layer_1/attention/self/key/bias
Š
Mbert/encoder/layer_1/attention/self/key/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp,bert/encoder/layer_1/attention/self/key/bias*
_output_shapes
: 
Â
3bert/encoder/layer_1/attention/self/key/bias/AssignAssignVariableOp,bert/encoder/layer_1/attention/self/key/bias>bert/encoder/layer_1/attention/self/key/bias/Initializer/zeros*
dtype0
Ş
@bert/encoder/layer_1/attention/self/key/bias/Read/ReadVariableOpReadVariableOp,bert/encoder/layer_1/attention/self/key/bias*
_output_shapes	
:*
dtype0
Ş
@bert/encoder/layer_1/attention/self/key/Reshape_1/ReadVariableOpReadVariableOp,bert/encoder/layer_1/attention/self/key/bias*
_output_shapes	
:*
dtype0

7bert/encoder/layer_1/attention/self/key/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   
ŕ
1bert/encoder/layer_1/attention/self/key/Reshape_1Reshape@bert/encoder/layer_1/attention/self/key/Reshape_1/ReadVariableOp7bert/encoder/layer_1/attention/self/key/Reshape_1/shape*
T0*
_output_shapes

:@

5bert/encoder/layer_1/attention/self/key/einsum/EinsumEinsum5bert/encoder/layer_0/output/LayerNorm/batchnorm/add_1/bert/encoder/layer_1/attention/self/key/Reshape*
N*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
equationabc,cde->abde
Ů
+bert/encoder/layer_1/attention/self/key/addAddV25bert/encoder/layer_1/attention/self/key/einsum/Einsum1bert/encoder/layer_1/attention/self/key/Reshape_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@

+bert/encoder/layer_1/attention/self/Shape_4Shape5bert/encoder/layer_0/output/LayerNorm/batchnorm/add_1*
T0*
_output_shapes
:

9bert/encoder/layer_1/attention/self/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB: 

;bert/encoder/layer_1/attention/self/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:

;bert/encoder/layer_1/attention/self/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
ë
3bert/encoder/layer_1/attention/self/strided_slice_4StridedSlice+bert/encoder/layer_1/attention/self/Shape_49bert/encoder/layer_1/attention/self/strided_slice_4/stack;bert/encoder/layer_1/attention/self/strided_slice_4/stack_1;bert/encoder/layer_1/attention/self/strided_slice_4/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
é
Sbert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal/shapeConst*C
_class9
75loc:@bert/encoder/layer_1/attention/self/value/kernel*
_output_shapes
:*
dtype0*
valueB"      
Ü
Rbert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal/meanConst*C
_class9
75loc:@bert/encoder/layer_1/attention/self/value/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
Ţ
Tbert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal/stddevConst*C
_class9
75loc:@bert/encoder/layer_1/attention/self/value/kernel*
_output_shapes
: *
dtype0*
valueB
 *
×Ł<
Ý
]bert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalSbert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal/shape*
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/self/value/kernel* 
_output_shapes
:
*
dtype0*
seed˛*
seed2
ý
Qbert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal/mulMul]bert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal/TruncatedNormalTbert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal/stddev*
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/self/value/kernel* 
_output_shapes
:

ë
Mbert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normalAddQbert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal/mulRbert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal/mean*
T0*C
_class9
75loc:@bert/encoder/layer_1/attention/self/value/kernel* 
_output_shapes
:


0bert/encoder/layer_1/attention/self/value/kernelVarHandleOp*C
_class9
75loc:@bert/encoder/layer_1/attention/self/value/kernel*
_output_shapes
: *
dtype0*
shape:
*A
shared_name20bert/encoder/layer_1/attention/self/value/kernel
ą
Qbert/encoder/layer_1/attention/self/value/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp0bert/encoder/layer_1/attention/self/value/kernel*
_output_shapes
: 
Ů
7bert/encoder/layer_1/attention/self/value/kernel/AssignAssignVariableOp0bert/encoder/layer_1/attention/self/value/kernelMbert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal*
dtype0
ˇ
Dbert/encoder/layer_1/attention/self/value/kernel/Read/ReadVariableOpReadVariableOp0bert/encoder/layer_1/attention/self/value/kernel* 
_output_shapes
:
*
dtype0
ł
@bert/encoder/layer_1/attention/self/value/Reshape/ReadVariableOpReadVariableOp0bert/encoder/layer_1/attention/self/value/kernel* 
_output_shapes
:
*
dtype0

7bert/encoder/layer_1/attention/self/value/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"      @   
ĺ
1bert/encoder/layer_1/attention/self/value/ReshapeReshape@bert/encoder/layer_1/attention/self/value/Reshape/ReadVariableOp7bert/encoder/layer_1/attention/self/value/Reshape/shape*
T0*#
_output_shapes
:@
Ň
@bert/encoder/layer_1/attention/self/value/bias/Initializer/zerosConst*A
_class7
53loc:@bert/encoder/layer_1/attention/self/value/bias*
_output_shapes	
:*
dtype0*
valueB*    
ř
.bert/encoder/layer_1/attention/self/value/biasVarHandleOp*A
_class7
53loc:@bert/encoder/layer_1/attention/self/value/bias*
_output_shapes
: *
dtype0*
shape:*?
shared_name0.bert/encoder/layer_1/attention/self/value/bias
­
Obert/encoder/layer_1/attention/self/value/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp.bert/encoder/layer_1/attention/self/value/bias*
_output_shapes
: 
Č
5bert/encoder/layer_1/attention/self/value/bias/AssignAssignVariableOp.bert/encoder/layer_1/attention/self/value/bias@bert/encoder/layer_1/attention/self/value/bias/Initializer/zeros*
dtype0
Ž
Bbert/encoder/layer_1/attention/self/value/bias/Read/ReadVariableOpReadVariableOp.bert/encoder/layer_1/attention/self/value/bias*
_output_shapes	
:*
dtype0
Ž
Bbert/encoder/layer_1/attention/self/value/Reshape_1/ReadVariableOpReadVariableOp.bert/encoder/layer_1/attention/self/value/bias*
_output_shapes	
:*
dtype0

9bert/encoder/layer_1/attention/self/value/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   
ć
3bert/encoder/layer_1/attention/self/value/Reshape_1ReshapeBbert/encoder/layer_1/attention/self/value/Reshape_1/ReadVariableOp9bert/encoder/layer_1/attention/self/value/Reshape_1/shape*
T0*
_output_shapes

:@

7bert/encoder/layer_1/attention/self/value/einsum/EinsumEinsum5bert/encoder/layer_0/output/LayerNorm/batchnorm/add_11bert/encoder/layer_1/attention/self/value/Reshape*
N*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
equationabc,cde->abde
ß
-bert/encoder/layer_1/attention/self/value/addAddV27bert/encoder/layer_1/attention/self/value/einsum/Einsum3bert/encoder/layer_1/attention/self/value/Reshape_1*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@
ű
1bert/encoder/layer_1/attention/self/einsum/EinsumEinsum+bert/encoder/layer_1/attention/self/key/add-bert/encoder/layer_1/attention/self/query/add*
N*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙*
equationBTNH,BFNH->BNFT
n
)bert/encoder/layer_1/attention/self/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >
Č
'bert/encoder/layer_1/attention/self/MulMul1bert/encoder/layer_1/attention/self/einsum/Einsum)bert/encoder/layer_1/attention/self/Mul/y*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
|
2bert/encoder/layer_1/attention/self/ExpandDims/dimConst*
_output_shapes
:*
dtype0*
valueB:
ž
.bert/encoder/layer_1/attention/self/ExpandDims
ExpandDimsbert/encoder/mul2bert/encoder/layer_1/attention/self/ExpandDims/dim*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
n
)bert/encoder/layer_1/attention/self/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
Ĺ
'bert/encoder/layer_1/attention/self/subSub)bert/encoder/layer_1/attention/self/sub/x.bert/encoder/layer_1/attention/self/ExpandDims*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
+bert/encoder/layer_1/attention/self/mul_1/yConst*
_output_shapes
: *
dtype0*
valueB
 * @Ć
Â
)bert/encoder/layer_1/attention/self/mul_1Mul'bert/encoder/layer_1/attention/self/sub+bert/encoder/layer_1/attention/self/mul_1/y*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ŕ
'bert/encoder/layer_1/attention/self/addAddV2'bert/encoder/layer_1/attention/self/Mul)bert/encoder/layer_1/attention/self/mul_1*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙

+bert/encoder/layer_1/attention/self/SoftmaxSoftmax'bert/encoder/layer_1/attention/self/add*
T0*1
_output_shapes
:˙˙˙˙˙˙˙˙˙
ü
3bert/encoder/layer_1/attention/self/einsum_1/EinsumEinsum+bert/encoder/layer_1/attention/self/Softmax-bert/encoder/layer_1/attention/self/value/add*
N*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
equationBNFT,BTNH->BFNH
í
Ubert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal/shapeConst*E
_class;
97loc:@bert/encoder/layer_1/attention/output/dense/kernel*
_output_shapes
:*
dtype0*
valueB"      
ŕ
Tbert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal/meanConst*E
_class;
97loc:@bert/encoder/layer_1/attention/output/dense/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
â
Vbert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal/stddevConst*E
_class;
97loc:@bert/encoder/layer_1/attention/output/dense/kernel*
_output_shapes
: *
dtype0*
valueB
 *
×Ł<
ă
_bert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalUbert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal/shape*
T0*E
_class;
97loc:@bert/encoder/layer_1/attention/output/dense/kernel* 
_output_shapes
:
*
dtype0*
seed˛*
seed2

Sbert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal/mulMul_bert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalVbert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*E
_class;
97loc:@bert/encoder/layer_1/attention/output/dense/kernel* 
_output_shapes
:

ó
Obert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normalAddSbert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal/mulTbert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal/mean*
T0*E
_class;
97loc:@bert/encoder/layer_1/attention/output/dense/kernel* 
_output_shapes
:


2bert/encoder/layer_1/attention/output/dense/kernelVarHandleOp*E
_class;
97loc:@bert/encoder/layer_1/attention/output/dense/kernel*
_output_shapes
: *
dtype0*
shape:
*C
shared_name42bert/encoder/layer_1/attention/output/dense/kernel
ľ
Sbert/encoder/layer_1/attention/output/dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp2bert/encoder/layer_1/attention/output/dense/kernel*
_output_shapes
: 
ß
9bert/encoder/layer_1/attention/output/dense/kernel/AssignAssignVariableOp2bert/encoder/layer_1/attention/output/dense/kernelObert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal*
dtype0
ť
Fbert/encoder/layer_1/attention/output/dense/kernel/Read/ReadVariableOpReadVariableOp2bert/encoder/layer_1/attention/output/dense/kernel* 
_output_shapes
:
*
dtype0
ˇ
Bbert/encoder/layer_1/attention/output/dense/Reshape/ReadVariableOpReadVariableOp2bert/encoder/layer_1/attention/output/dense/kernel* 
_output_shapes
:
*
dtype0

9bert/encoder/layer_1/attention/output/dense/Reshape/shapeConst*
_output_shapes
:*
dtype0*!
valueB"   @      
ë
3bert/encoder/layer_1/attention/output/dense/ReshapeReshapeBbert/encoder/layer_1/attention/output/dense/Reshape/ReadVariableOp9bert/encoder/layer_1/attention/output/dense/Reshape/shape*
T0*#
_output_shapes
:@
Ö
Bbert/encoder/layer_1/attention/output/dense/bias/Initializer/zerosConst*C
_class9
75loc:@bert/encoder/layer_1/attention/output/dense/bias*
_output_shapes	
:*
dtype0*
valueB*    
ţ
0bert/encoder/layer_1/attention/output/dense/biasVarHandleOp*C
_class9
75loc:@bert/encoder/layer_1/attention/output/dense/bias*
_output_shapes
: *
dtype0*
shape:*A
shared_name20bert/encoder/layer_1/attention/output/dense/bias
ą
Qbert/encoder/layer_1/attention/output/dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp0bert/encoder/layer_1/attention/output/dense/bias*
_output_shapes
: 
Î
7bert/encoder/layer_1/attention/output/dense/bias/AssignAssignVariableOp0bert/encoder/layer_1/attention/output/dense/biasBbert/encoder/layer_1/attention/output/dense/bias/Initializer/zeros*
dtype0
˛
Dbert/encoder/layer_1/attention/output/dense/bias/Read/ReadVariableOpReadVariableOp0bert/encoder/layer_1/attention/output/dense/bias*
_output_shapes	
:*
dtype0

3bert/encoder/layer_1/attention/output/einsum/EinsumEinsum3bert/encoder/layer_1/attention/self/einsum_1/Einsum3bert/encoder/layer_1/attention/output/dense/Reshape*
N*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙*
equationBFNH,NHD->BFD
Ś
8bert/encoder/layer_1/attention/output/add/ReadVariableOpReadVariableOp0bert/encoder/layer_1/attention/output/dense/bias*
_output_shapes	
:*
dtype0
Ů
)bert/encoder/layer_1/attention/output/addAddV23bert/encoder/layer_1/attention/output/einsum/Einsum8bert/encoder/layer_1/attention/output/add/ReadVariableOp*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
Î
+bert/encoder/layer_1/attention/output/add_1AddV2)bert/encoder/layer_1/attention/output/add5bert/encoder/layer_0/output/LayerNorm/batchnorm/add_1*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ţ
Fbert/encoder/layer_1/attention/output/LayerNorm/beta/Initializer/zerosConst*G
_class=
;9loc:@bert/encoder/layer_1/attention/output/LayerNorm/beta*
_output_shapes	
:*
dtype0*
valueB*    

4bert/encoder/layer_1/attention/output/LayerNorm/betaVarHandleOp*G
_class=
;9loc:@bert/encoder/layer_1/attention/output/LayerNorm/beta*
_output_shapes
: *
dtype0*
shape:*E
shared_name64bert/encoder/layer_1/attention/output/LayerNorm/beta
š
Ubert/encoder/layer_1/attention/output/LayerNorm/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp4bert/encoder/layer_1/attention/output/LayerNorm/beta*
_output_shapes
: 
Ú
;bert/encoder/layer_1/attention/output/LayerNorm/beta/AssignAssignVariableOp4bert/encoder/layer_1/attention/output/LayerNorm/betaFbert/encoder/layer_1/attention/output/LayerNorm/beta/Initializer/zeros*
dtype0
ş
Hbert/encoder/layer_1/attention/output/LayerNorm/beta/Read/ReadVariableOpReadVariableOp4bert/encoder/layer_1/attention/output/LayerNorm/beta*
_output_shapes	
:*
dtype0
ß
Fbert/encoder/layer_1/attention/output/LayerNorm/gamma/Initializer/onesConst*H
_class>
<:loc:@bert/encoder/layer_1/attention/output/LayerNorm/gamma*
_output_shapes	
:*
dtype0*
valueB*  ?

5bert/encoder/layer_1/attention/output/LayerNorm/gammaVarHandleOp*H
_class>
<:loc:@bert/encoder/layer_1/attention/output/LayerNorm/gamma*
_output_shapes
: *
dtype0*
shape:*F
shared_name75bert/encoder/layer_1/attention/output/LayerNorm/gamma
ť
Vbert/encoder/layer_1/attention/output/LayerNorm/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp5bert/encoder/layer_1/attention/output/LayerNorm/gamma*
_output_shapes
: 
Ü
<bert/encoder/layer_1/attention/output/LayerNorm/gamma/AssignAssignVariableOp5bert/encoder/layer_1/attention/output/LayerNorm/gammaFbert/encoder/layer_1/attention/output/LayerNorm/gamma/Initializer/ones*
dtype0
ź
Ibert/encoder/layer_1/attention/output/LayerNorm/gamma/Read/ReadVariableOpReadVariableOp5bert/encoder/layer_1/attention/output/LayerNorm/gamma*
_output_shapes	
:*
dtype0

Nbert/encoder/layer_1/attention/output/LayerNorm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:

<bert/encoder/layer_1/attention/output/LayerNorm/moments/meanMean+bert/encoder/layer_1/attention/output/add_1Nbert/encoder/layer_1/attention/output/LayerNorm/moments/mean/reduction_indices*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(
É
Dbert/encoder/layer_1/attention/output/LayerNorm/moments/StopGradientStopGradient<bert/encoder/layer_1/attention/output/LayerNorm/moments/mean*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙

Ibert/encoder/layer_1/attention/output/LayerNorm/moments/SquaredDifferenceSquaredDifference+bert/encoder/layer_1/attention/output/add_1Dbert/encoder/layer_1/attention/output/LayerNorm/moments/StopGradient*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙

Rbert/encoder/layer_1/attention/output/LayerNorm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
Ż
@bert/encoder/layer_1/attention/output/LayerNorm/moments/varianceMeanIbert/encoder/layer_1/attention/output/LayerNorm/moments/SquaredDifferenceRbert/encoder/layer_1/attention/output/LayerNorm/moments/variance/reduction_indices*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(

?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ěź+

=bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/addAddV2@bert/encoder/layer_1/attention/output/LayerNorm/moments/variance?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/add/y*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
ž
?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/RsqrtRsqrt=bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/add*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
ż
Lbert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mul/ReadVariableOpReadVariableOp5bert/encoder/layer_1/attention/output/LayerNorm/gamma*
_output_shapes	
:*
dtype0

=bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mulMul?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/RsqrtLbert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mul/ReadVariableOp*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
ę
?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mul_1Mul+bert/encoder/layer_1/attention/output/add_1=bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mul*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
ű
?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mul_2Mul<bert/encoder/layer_1/attention/output/LayerNorm/moments/mean=bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mul*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
ş
Hbert/encoder/layer_1/attention/output/LayerNorm/batchnorm/ReadVariableOpReadVariableOp4bert/encoder/layer_1/attention/output/LayerNorm/beta*
_output_shapes	
:*
dtype0

=bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/subSubHbert/encoder/layer_1/attention/output/LayerNorm/batchnorm/ReadVariableOp?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mul_2*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙

?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/add_1AddV2?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/mul_1=bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/sub*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙

'bert/encoder/layer_1/intermediate/ShapeShape?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/add_1*
T0*
_output_shapes
:

5bert/encoder/layer_1/intermediate/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 

7bert/encoder/layer_1/intermediate/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:

7bert/encoder/layer_1/intermediate/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
×
/bert/encoder/layer_1/intermediate/strided_sliceStridedSlice'bert/encoder/layer_1/intermediate/Shape5bert/encoder/layer_1/intermediate/strided_slice/stack7bert/encoder/layer_1/intermediate/strided_slice/stack_17bert/encoder/layer_1/intermediate/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
ĺ
Qbert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal/shapeConst*A
_class7
53loc:@bert/encoder/layer_1/intermediate/dense/kernel*
_output_shapes
:*
dtype0*
valueB"      
Ř
Pbert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal/meanConst*A
_class7
53loc:@bert/encoder/layer_1/intermediate/dense/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
Ú
Rbert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal/stddevConst*A
_class7
53loc:@bert/encoder/layer_1/intermediate/dense/kernel*
_output_shapes
: *
dtype0*
valueB
 *
×Ł<
×
[bert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalQbert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal/shape*
T0*A
_class7
53loc:@bert/encoder/layer_1/intermediate/dense/kernel* 
_output_shapes
:
*
dtype0*
seed˛*
seed2
ő
Obert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal/mulMul[bert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal/TruncatedNormalRbert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal/stddev*
T0*A
_class7
53loc:@bert/encoder/layer_1/intermediate/dense/kernel* 
_output_shapes
:

ă
Kbert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normalAddObert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal/mulPbert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal/mean*
T0*A
_class7
53loc:@bert/encoder/layer_1/intermediate/dense/kernel* 
_output_shapes
:

ý
.bert/encoder/layer_1/intermediate/dense/kernelVarHandleOp*A
_class7
53loc:@bert/encoder/layer_1/intermediate/dense/kernel*
_output_shapes
: *
dtype0*
shape:
*?
shared_name0.bert/encoder/layer_1/intermediate/dense/kernel
­
Obert/encoder/layer_1/intermediate/dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp.bert/encoder/layer_1/intermediate/dense/kernel*
_output_shapes
: 
Ó
5bert/encoder/layer_1/intermediate/dense/kernel/AssignAssignVariableOp.bert/encoder/layer_1/intermediate/dense/kernelKbert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal*
dtype0
ł
Bbert/encoder/layer_1/intermediate/dense/kernel/Read/ReadVariableOpReadVariableOp.bert/encoder/layer_1/intermediate/dense/kernel* 
_output_shapes
:
*
dtype0
Î
>bert/encoder/layer_1/intermediate/dense/bias/Initializer/zerosConst*?
_class5
31loc:@bert/encoder/layer_1/intermediate/dense/bias*
_output_shapes	
:*
dtype0*
valueB*    
ň
,bert/encoder/layer_1/intermediate/dense/biasVarHandleOp*?
_class5
31loc:@bert/encoder/layer_1/intermediate/dense/bias*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,bert/encoder/layer_1/intermediate/dense/bias
Š
Mbert/encoder/layer_1/intermediate/dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp,bert/encoder/layer_1/intermediate/dense/bias*
_output_shapes
: 
Â
3bert/encoder/layer_1/intermediate/dense/bias/AssignAssignVariableOp,bert/encoder/layer_1/intermediate/dense/bias>bert/encoder/layer_1/intermediate/dense/bias/Initializer/zeros*
dtype0
Ş
@bert/encoder/layer_1/intermediate/dense/bias/Read/ReadVariableOpReadVariableOp,bert/encoder/layer_1/intermediate/dense/bias*
_output_shapes	
:*
dtype0
Ż
>bert/encoder/layer_1/intermediate/einsum/Einsum/ReadVariableOpReadVariableOp.bert/encoder/layer_1/intermediate/dense/kernel* 
_output_shapes
:
*
dtype0

/bert/encoder/layer_1/intermediate/einsum/EinsumEinsum?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/add_1>bert/encoder/layer_1/intermediate/einsum/Einsum/ReadVariableOp*
N*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙*
equationabc,cd->abd

4bert/encoder/layer_1/intermediate/add/ReadVariableOpReadVariableOp,bert/encoder/layer_1/intermediate/dense/bias*
_output_shapes	
:*
dtype0
Í
%bert/encoder/layer_1/intermediate/addAddV2/bert/encoder/layer_1/intermediate/einsum/Einsum4bert/encoder/layer_1/intermediate/add/ReadVariableOp*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
l
'bert/encoder/layer_1/intermediate/Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *  @@
´
%bert/encoder/layer_1/intermediate/PowPow%bert/encoder/layer_1/intermediate/add'bert/encoder/layer_1/intermediate/Pow/y*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
l
'bert/encoder/layer_1/intermediate/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *'7=
´
%bert/encoder/layer_1/intermediate/mulMul'bert/encoder/layer_1/intermediate/mul/x%bert/encoder/layer_1/intermediate/Pow*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
ś
'bert/encoder/layer_1/intermediate/add_1AddV2%bert/encoder/layer_1/intermediate/add%bert/encoder/layer_1/intermediate/mul*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
n
)bert/encoder/layer_1/intermediate/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 **BL?
ş
'bert/encoder/layer_1/intermediate/mul_1Mul)bert/encoder/layer_1/intermediate/mul_1/x'bert/encoder/layer_1/intermediate/add_1*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙

&bert/encoder/layer_1/intermediate/TanhTanh'bert/encoder/layer_1/intermediate/mul_1*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
n
)bert/encoder/layer_1/intermediate/add_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
ť
'bert/encoder/layer_1/intermediate/add_2AddV2)bert/encoder/layer_1/intermediate/add_2/x&bert/encoder/layer_1/intermediate/Tanh*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
n
)bert/encoder/layer_1/intermediate/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
ş
'bert/encoder/layer_1/intermediate/mul_2Mul)bert/encoder/layer_1/intermediate/mul_2/x'bert/encoder/layer_1/intermediate/add_2*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
ś
'bert/encoder/layer_1/intermediate/mul_3Mul%bert/encoder/layer_1/intermediate/add'bert/encoder/layer_1/intermediate/mul_2*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
x
!bert/encoder/layer_1/output/ShapeShape'bert/encoder/layer_1/intermediate/mul_3*
T0*
_output_shapes
:
y
/bert/encoder/layer_1/output/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
{
1bert/encoder/layer_1/output/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
{
1bert/encoder/layer_1/output/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
š
)bert/encoder/layer_1/output/strided_sliceStridedSlice!bert/encoder/layer_1/output/Shape/bert/encoder/layer_1/output/strided_slice/stack1bert/encoder/layer_1/output/strided_slice/stack_11bert/encoder/layer_1/output/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Ů
Kbert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal/shapeConst*;
_class1
/-loc:@bert/encoder/layer_1/output/dense/kernel*
_output_shapes
:*
dtype0*
valueB"      
Ě
Jbert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal/meanConst*;
_class1
/-loc:@bert/encoder/layer_1/output/dense/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
Î
Lbert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal/stddevConst*;
_class1
/-loc:@bert/encoder/layer_1/output/dense/kernel*
_output_shapes
: *
dtype0*
valueB
 *
×Ł<
Ĺ
Ubert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormalKbert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal/shape*
T0*;
_class1
/-loc:@bert/encoder/layer_1/output/dense/kernel* 
_output_shapes
:
*
dtype0*
seed˛*
seed2
Ý
Ibert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal/mulMulUbert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal/TruncatedNormalLbert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal/stddev*
T0*;
_class1
/-loc:@bert/encoder/layer_1/output/dense/kernel* 
_output_shapes
:

Ë
Ebert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normalAddIbert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal/mulJbert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal/mean*
T0*;
_class1
/-loc:@bert/encoder/layer_1/output/dense/kernel* 
_output_shapes
:

ë
(bert/encoder/layer_1/output/dense/kernelVarHandleOp*;
_class1
/-loc:@bert/encoder/layer_1/output/dense/kernel*
_output_shapes
: *
dtype0*
shape:
*9
shared_name*(bert/encoder/layer_1/output/dense/kernel
Ą
Ibert/encoder/layer_1/output/dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOp(bert/encoder/layer_1/output/dense/kernel*
_output_shapes
: 
Á
/bert/encoder/layer_1/output/dense/kernel/AssignAssignVariableOp(bert/encoder/layer_1/output/dense/kernelEbert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal*
dtype0
§
<bert/encoder/layer_1/output/dense/kernel/Read/ReadVariableOpReadVariableOp(bert/encoder/layer_1/output/dense/kernel* 
_output_shapes
:
*
dtype0
Â
8bert/encoder/layer_1/output/dense/bias/Initializer/zerosConst*9
_class/
-+loc:@bert/encoder/layer_1/output/dense/bias*
_output_shapes	
:*
dtype0*
valueB*    
ŕ
&bert/encoder/layer_1/output/dense/biasVarHandleOp*9
_class/
-+loc:@bert/encoder/layer_1/output/dense/bias*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&bert/encoder/layer_1/output/dense/bias

Gbert/encoder/layer_1/output/dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp&bert/encoder/layer_1/output/dense/bias*
_output_shapes
: 
°
-bert/encoder/layer_1/output/dense/bias/AssignAssignVariableOp&bert/encoder/layer_1/output/dense/bias8bert/encoder/layer_1/output/dense/bias/Initializer/zeros*
dtype0

:bert/encoder/layer_1/output/dense/bias/Read/ReadVariableOpReadVariableOp&bert/encoder/layer_1/output/dense/bias*
_output_shapes	
:*
dtype0
Ł
8bert/encoder/layer_1/output/einsum/Einsum/ReadVariableOpReadVariableOp(bert/encoder/layer_1/output/dense/kernel* 
_output_shapes
:
*
dtype0
ň
)bert/encoder/layer_1/output/einsum/EinsumEinsum'bert/encoder/layer_1/intermediate/mul_38bert/encoder/layer_1/output/einsum/Einsum/ReadVariableOp*
N*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙*
equationabc,cd->abd

.bert/encoder/layer_1/output/add/ReadVariableOpReadVariableOp&bert/encoder/layer_1/output/dense/bias*
_output_shapes	
:*
dtype0
ť
bert/encoder/layer_1/output/addAddV2)bert/encoder/layer_1/output/einsum/Einsum.bert/encoder/layer_1/output/add/ReadVariableOp*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ä
!bert/encoder/layer_1/output/add_1AddV2bert/encoder/layer_1/output/add?bert/encoder/layer_1/attention/output/LayerNorm/batchnorm/add_1*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ę
<bert/encoder/layer_1/output/LayerNorm/beta/Initializer/zerosConst*=
_class3
1/loc:@bert/encoder/layer_1/output/LayerNorm/beta*
_output_shapes	
:*
dtype0*
valueB*    
ě
*bert/encoder/layer_1/output/LayerNorm/betaVarHandleOp*=
_class3
1/loc:@bert/encoder/layer_1/output/LayerNorm/beta*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*bert/encoder/layer_1/output/LayerNorm/beta
Ľ
Kbert/encoder/layer_1/output/LayerNorm/beta/IsInitialized/VarIsInitializedOpVarIsInitializedOp*bert/encoder/layer_1/output/LayerNorm/beta*
_output_shapes
: 
ź
1bert/encoder/layer_1/output/LayerNorm/beta/AssignAssignVariableOp*bert/encoder/layer_1/output/LayerNorm/beta<bert/encoder/layer_1/output/LayerNorm/beta/Initializer/zeros*
dtype0
Ś
>bert/encoder/layer_1/output/LayerNorm/beta/Read/ReadVariableOpReadVariableOp*bert/encoder/layer_1/output/LayerNorm/beta*
_output_shapes	
:*
dtype0
Ë
<bert/encoder/layer_1/output/LayerNorm/gamma/Initializer/onesConst*>
_class4
20loc:@bert/encoder/layer_1/output/LayerNorm/gamma*
_output_shapes	
:*
dtype0*
valueB*  ?
ď
+bert/encoder/layer_1/output/LayerNorm/gammaVarHandleOp*>
_class4
20loc:@bert/encoder/layer_1/output/LayerNorm/gamma*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+bert/encoder/layer_1/output/LayerNorm/gamma
§
Lbert/encoder/layer_1/output/LayerNorm/gamma/IsInitialized/VarIsInitializedOpVarIsInitializedOp+bert/encoder/layer_1/output/LayerNorm/gamma*
_output_shapes
: 
ž
2bert/encoder/layer_1/output/LayerNorm/gamma/AssignAssignVariableOp+bert/encoder/layer_1/output/LayerNorm/gamma<bert/encoder/layer_1/output/LayerNorm/gamma/Initializer/ones*
dtype0
¨
?bert/encoder/layer_1/output/LayerNorm/gamma/Read/ReadVariableOpReadVariableOp+bert/encoder/layer_1/output/LayerNorm/gamma*
_output_shapes	
:*
dtype0

Dbert/encoder/layer_1/output/LayerNorm/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:
ë
2bert/encoder/layer_1/output/LayerNorm/moments/meanMean!bert/encoder/layer_1/output/add_1Dbert/encoder/layer_1/output/LayerNorm/moments/mean/reduction_indices*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(
ľ
:bert/encoder/layer_1/output/LayerNorm/moments/StopGradientStopGradient2bert/encoder/layer_1/output/LayerNorm/moments/mean*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
ë
?bert/encoder/layer_1/output/LayerNorm/moments/SquaredDifferenceSquaredDifference!bert/encoder/layer_1/output/add_1:bert/encoder/layer_1/output/LayerNorm/moments/StopGradient*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙

Hbert/encoder/layer_1/output/LayerNorm/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:

6bert/encoder/layer_1/output/LayerNorm/moments/varianceMean?bert/encoder/layer_1/output/LayerNorm/moments/SquaredDifferenceHbert/encoder/layer_1/output/LayerNorm/moments/variance/reduction_indices*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	keep_dims(
z
5bert/encoder/layer_1/output/LayerNorm/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ěź+
â
3bert/encoder/layer_1/output/LayerNorm/batchnorm/addAddV26bert/encoder/layer_1/output/LayerNorm/moments/variance5bert/encoder/layer_1/output/LayerNorm/batchnorm/add/y*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ş
5bert/encoder/layer_1/output/LayerNorm/batchnorm/RsqrtRsqrt3bert/encoder/layer_1/output/LayerNorm/batchnorm/add*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ť
Bbert/encoder/layer_1/output/LayerNorm/batchnorm/mul/ReadVariableOpReadVariableOp+bert/encoder/layer_1/output/LayerNorm/gamma*
_output_shapes	
:*
dtype0
í
3bert/encoder/layer_1/output/LayerNorm/batchnorm/mulMul5bert/encoder/layer_1/output/LayerNorm/batchnorm/RsqrtBbert/encoder/layer_1/output/LayerNorm/batchnorm/mul/ReadVariableOp*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ě
5bert/encoder/layer_1/output/LayerNorm/batchnorm/mul_1Mul!bert/encoder/layer_1/output/add_13bert/encoder/layer_1/output/LayerNorm/batchnorm/mul*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ý
5bert/encoder/layer_1/output/LayerNorm/batchnorm/mul_2Mul2bert/encoder/layer_1/output/LayerNorm/moments/mean3bert/encoder/layer_1/output/LayerNorm/batchnorm/mul*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ś
>bert/encoder/layer_1/output/LayerNorm/batchnorm/ReadVariableOpReadVariableOp*bert/encoder/layer_1/output/LayerNorm/beta*
_output_shapes	
:*
dtype0
é
3bert/encoder/layer_1/output/LayerNorm/batchnorm/subSub>bert/encoder/layer_1/output/LayerNorm/batchnorm/ReadVariableOp5bert/encoder/layer_1/output/LayerNorm/batchnorm/mul_2*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
â
5bert/encoder/layer_1/output/LayerNorm/batchnorm/add_1AddV25bert/encoder/layer_1/output/LayerNorm/batchnorm/mul_13bert/encoder/layer_1/output/LayerNorm/batchnorm/sub*
T0*-
_output_shapes
:˙˙˙˙˙˙˙˙˙
t
bert/pooler/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"            
v
!bert/pooler/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           
v
!bert/pooler/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         
­
bert/pooler/strided_sliceStridedSlice5bert/encoder/layer_1/output/LayerNorm/batchnorm/add_1bert/pooler/strided_slice/stack!bert/pooler/strided_slice/stack_1!bert/pooler/strided_slice/stack_2*
Index0*
T0*,
_output_shapes
:˙˙˙˙˙˙˙˙˙*

begin_mask*
end_mask

bert/pooler/SqueezeSqueezebert/pooler/strided_slice*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
squeeze_dims

š
;bert/pooler/dense/kernel/Initializer/truncated_normal/shapeConst*+
_class!
loc:@bert/pooler/dense/kernel*
_output_shapes
:*
dtype0*
valueB"      
Ź
:bert/pooler/dense/kernel/Initializer/truncated_normal/meanConst*+
_class!
loc:@bert/pooler/dense/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
Ž
<bert/pooler/dense/kernel/Initializer/truncated_normal/stddevConst*+
_class!
loc:@bert/pooler/dense/kernel*
_output_shapes
: *
dtype0*
valueB
 *
×Ł<

Ebert/pooler/dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal;bert/pooler/dense/kernel/Initializer/truncated_normal/shape*
T0*+
_class!
loc:@bert/pooler/dense/kernel* 
_output_shapes
:
*
dtype0*
seed˛*
seed2

9bert/pooler/dense/kernel/Initializer/truncated_normal/mulMulEbert/pooler/dense/kernel/Initializer/truncated_normal/TruncatedNormal<bert/pooler/dense/kernel/Initializer/truncated_normal/stddev*
T0*+
_class!
loc:@bert/pooler/dense/kernel* 
_output_shapes
:


5bert/pooler/dense/kernel/Initializer/truncated_normalAdd9bert/pooler/dense/kernel/Initializer/truncated_normal/mul:bert/pooler/dense/kernel/Initializer/truncated_normal/mean*
T0*+
_class!
loc:@bert/pooler/dense/kernel* 
_output_shapes
:

ť
bert/pooler/dense/kernelVarHandleOp*+
_class!
loc:@bert/pooler/dense/kernel*
_output_shapes
: *
dtype0*
shape:
*)
shared_namebert/pooler/dense/kernel

9bert/pooler/dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpbert/pooler/dense/kernel*
_output_shapes
: 

bert/pooler/dense/kernel/AssignAssignVariableOpbert/pooler/dense/kernel5bert/pooler/dense/kernel/Initializer/truncated_normal*
dtype0

,bert/pooler/dense/kernel/Read/ReadVariableOpReadVariableOpbert/pooler/dense/kernel* 
_output_shapes
:
*
dtype0
˘
(bert/pooler/dense/bias/Initializer/zerosConst*)
_class
loc:@bert/pooler/dense/bias*
_output_shapes	
:*
dtype0*
valueB*    
°
bert/pooler/dense/biasVarHandleOp*)
_class
loc:@bert/pooler/dense/bias*
_output_shapes
: *
dtype0*
shape:*'
shared_namebert/pooler/dense/bias
}
7bert/pooler/dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOpbert/pooler/dense/bias*
_output_shapes
: 

bert/pooler/dense/bias/AssignAssignVariableOpbert/pooler/dense/bias(bert/pooler/dense/bias/Initializer/zeros*
dtype0
~
*bert/pooler/dense/bias/Read/ReadVariableOpReadVariableOpbert/pooler/dense/bias*
_output_shapes	
:*
dtype0

'bert/pooler/dense/MatMul/ReadVariableOpReadVariableOpbert/pooler/dense/kernel* 
_output_shapes
:
*
dtype0

bert/pooler/dense/MatMulMatMulbert/pooler/Squeeze'bert/pooler/dense/MatMul/ReadVariableOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
|
(bert/pooler/dense/BiasAdd/ReadVariableOpReadVariableOpbert/pooler/dense/bias*
_output_shapes	
:*
dtype0

bert/pooler/dense/BiasAddBiasAddbert/pooler/dense/MatMul(bert/pooler/dense/BiasAdd/ReadVariableOp*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
l
bert/pooler/dense/TanhTanhbert/pooler/dense/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

-dense/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@dense/kernel*
_output_shapes
:*
dtype0*
valueB"      

+dense/kernel/Initializer/random_uniform/minConst*
_class
loc:@dense/kernel*
_output_shapes
: *
dtype0*
valueB
 *n×\ž

+dense/kernel/Initializer/random_uniform/maxConst*
_class
loc:@dense/kernel*
_output_shapes
: *
dtype0*
valueB
 *n×\>
č
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	*
dtype0*
seed˛*
seed2
Î
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 
á
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	
Ó
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	

dense/kernelVarHandleOp*
_class
loc:@dense/kernel*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense/kernel
i
-dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense/kernel*
_output_shapes
: 
k
dense/kernel/AssignAssignVariableOpdense/kernel'dense/kernel/Initializer/random_uniform*
dtype0
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	*
dtype0

dense/bias/Initializer/ConstConst*
_class
loc:@dense/bias*
_output_shapes
:*
dtype0*
valueB*>


dense/biasVarHandleOp*
_class
loc:@dense/bias*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
+dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp
dense/bias*
_output_shapes
: 
\
dense/bias/AssignAssignVariableOp
dense/biasdense/bias/Initializer/Const*
dtype0
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
i
dense/MatMul/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	*
dtype0
}
dense/MatMulMatMulbert/pooler/dense/Tanhdense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
c
dense/BiasAdd/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
v
dense/BiasAddBiasAdddense/MatMuldense/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
f
SqueezeSqueezedense/BiasAdd*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
squeeze_dims

H
subSubSqueezezeros*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
J
Pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @
D
PowPowsubPow/y*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
a
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
J
MeanMeanPowMean/reduction_indices*
T0*
_output_shapes
: 

initNoOp

init_all_tablesNoOp

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
Y
save/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
dtype0*
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
dtype0*
shape: 
{
save/StaticRegexFullMatchStaticRegexFullMatch
save/Const"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*
a
save/Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part

save/Const_2Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_b86f92142ef243deaa191488697dd0ca/part
|
save/SelectSelectsave/StaticRegexFullMatchsave/Const_1save/Const_2"/device:CPU:**
T0*
_output_shapes
: 
f
save/StringJoin
StringJoin
save/Constsave/Select"/device:CPU:**
N*
_output_shapes
: 
Q
save/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
k
save/ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
ô
save/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*
valueB*Bbert/embeddings/LayerNorm/betaBbert/embeddings/LayerNorm/gammaB#bert/embeddings/position_embeddingsB%bert/embeddings/token_type_embeddingsBbert/embeddings/word_embeddingsB4bert/encoder/layer_0/attention/output/LayerNorm/betaB5bert/encoder/layer_0/attention/output/LayerNorm/gammaB0bert/encoder/layer_0/attention/output/dense/biasB2bert/encoder/layer_0/attention/output/dense/kernelB,bert/encoder/layer_0/attention/self/key/biasB.bert/encoder/layer_0/attention/self/key/kernelB.bert/encoder/layer_0/attention/self/query/biasB0bert/encoder/layer_0/attention/self/query/kernelB.bert/encoder/layer_0/attention/self/value/biasB0bert/encoder/layer_0/attention/self/value/kernelB,bert/encoder/layer_0/intermediate/dense/biasB.bert/encoder/layer_0/intermediate/dense/kernelB*bert/encoder/layer_0/output/LayerNorm/betaB+bert/encoder/layer_0/output/LayerNorm/gammaB&bert/encoder/layer_0/output/dense/biasB(bert/encoder/layer_0/output/dense/kernelB4bert/encoder/layer_1/attention/output/LayerNorm/betaB5bert/encoder/layer_1/attention/output/LayerNorm/gammaB0bert/encoder/layer_1/attention/output/dense/biasB2bert/encoder/layer_1/attention/output/dense/kernelB,bert/encoder/layer_1/attention/self/key/biasB.bert/encoder/layer_1/attention/self/key/kernelB.bert/encoder/layer_1/attention/self/query/biasB0bert/encoder/layer_1/attention/self/query/kernelB.bert/encoder/layer_1/attention/self/value/biasB0bert/encoder/layer_1/attention/self/value/kernelB,bert/encoder/layer_1/intermediate/dense/biasB.bert/encoder/layer_1/intermediate/dense/kernelB*bert/encoder/layer_1/output/LayerNorm/betaB+bert/encoder/layer_1/output/LayerNorm/gammaB&bert/encoder/layer_1/output/dense/biasB(bert/encoder/layer_1/output/dense/kernelBbert/pooler/dense/biasBbert/pooler/dense/kernelB
dense/biasBdense/kernelBglobal_step
Ć
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
÷
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices2bert/embeddings/LayerNorm/beta/Read/ReadVariableOp3bert/embeddings/LayerNorm/gamma/Read/ReadVariableOp7bert/embeddings/position_embeddings/Read/ReadVariableOp9bert/embeddings/token_type_embeddings/Read/ReadVariableOp3bert/embeddings/word_embeddings/Read/ReadVariableOpHbert/encoder/layer_0/attention/output/LayerNorm/beta/Read/ReadVariableOpIbert/encoder/layer_0/attention/output/LayerNorm/gamma/Read/ReadVariableOpDbert/encoder/layer_0/attention/output/dense/bias/Read/ReadVariableOpFbert/encoder/layer_0/attention/output/dense/kernel/Read/ReadVariableOp@bert/encoder/layer_0/attention/self/key/bias/Read/ReadVariableOpBbert/encoder/layer_0/attention/self/key/kernel/Read/ReadVariableOpBbert/encoder/layer_0/attention/self/query/bias/Read/ReadVariableOpDbert/encoder/layer_0/attention/self/query/kernel/Read/ReadVariableOpBbert/encoder/layer_0/attention/self/value/bias/Read/ReadVariableOpDbert/encoder/layer_0/attention/self/value/kernel/Read/ReadVariableOp@bert/encoder/layer_0/intermediate/dense/bias/Read/ReadVariableOpBbert/encoder/layer_0/intermediate/dense/kernel/Read/ReadVariableOp>bert/encoder/layer_0/output/LayerNorm/beta/Read/ReadVariableOp?bert/encoder/layer_0/output/LayerNorm/gamma/Read/ReadVariableOp:bert/encoder/layer_0/output/dense/bias/Read/ReadVariableOp<bert/encoder/layer_0/output/dense/kernel/Read/ReadVariableOpHbert/encoder/layer_1/attention/output/LayerNorm/beta/Read/ReadVariableOpIbert/encoder/layer_1/attention/output/LayerNorm/gamma/Read/ReadVariableOpDbert/encoder/layer_1/attention/output/dense/bias/Read/ReadVariableOpFbert/encoder/layer_1/attention/output/dense/kernel/Read/ReadVariableOp@bert/encoder/layer_1/attention/self/key/bias/Read/ReadVariableOpBbert/encoder/layer_1/attention/self/key/kernel/Read/ReadVariableOpBbert/encoder/layer_1/attention/self/query/bias/Read/ReadVariableOpDbert/encoder/layer_1/attention/self/query/kernel/Read/ReadVariableOpBbert/encoder/layer_1/attention/self/value/bias/Read/ReadVariableOpDbert/encoder/layer_1/attention/self/value/kernel/Read/ReadVariableOp@bert/encoder/layer_1/intermediate/dense/bias/Read/ReadVariableOpBbert/encoder/layer_1/intermediate/dense/kernel/Read/ReadVariableOp>bert/encoder/layer_1/output/LayerNorm/beta/Read/ReadVariableOp?bert/encoder/layer_1/output/LayerNorm/gamma/Read/ReadVariableOp:bert/encoder/layer_1/output/dense/bias/Read/ReadVariableOp<bert/encoder/layer_1/output/dense/kernel/Read/ReadVariableOp*bert/pooler/dense/bias/Read/ReadVariableOp,bert/pooler/dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpglobal_step/Read/ReadVariableOp"/device:CPU:0*8
dtypes.
,2*	
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
 
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
N*
T0*
_output_shapes
:
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0

save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
÷
save/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*
valueB*Bbert/embeddings/LayerNorm/betaBbert/embeddings/LayerNorm/gammaB#bert/embeddings/position_embeddingsB%bert/embeddings/token_type_embeddingsBbert/embeddings/word_embeddingsB4bert/encoder/layer_0/attention/output/LayerNorm/betaB5bert/encoder/layer_0/attention/output/LayerNorm/gammaB0bert/encoder/layer_0/attention/output/dense/biasB2bert/encoder/layer_0/attention/output/dense/kernelB,bert/encoder/layer_0/attention/self/key/biasB.bert/encoder/layer_0/attention/self/key/kernelB.bert/encoder/layer_0/attention/self/query/biasB0bert/encoder/layer_0/attention/self/query/kernelB.bert/encoder/layer_0/attention/self/value/biasB0bert/encoder/layer_0/attention/self/value/kernelB,bert/encoder/layer_0/intermediate/dense/biasB.bert/encoder/layer_0/intermediate/dense/kernelB*bert/encoder/layer_0/output/LayerNorm/betaB+bert/encoder/layer_0/output/LayerNorm/gammaB&bert/encoder/layer_0/output/dense/biasB(bert/encoder/layer_0/output/dense/kernelB4bert/encoder/layer_1/attention/output/LayerNorm/betaB5bert/encoder/layer_1/attention/output/LayerNorm/gammaB0bert/encoder/layer_1/attention/output/dense/biasB2bert/encoder/layer_1/attention/output/dense/kernelB,bert/encoder/layer_1/attention/self/key/biasB.bert/encoder/layer_1/attention/self/key/kernelB.bert/encoder/layer_1/attention/self/query/biasB0bert/encoder/layer_1/attention/self/query/kernelB.bert/encoder/layer_1/attention/self/value/biasB0bert/encoder/layer_1/attention/self/value/kernelB,bert/encoder/layer_1/intermediate/dense/biasB.bert/encoder/layer_1/intermediate/dense/kernelB*bert/encoder/layer_1/output/LayerNorm/betaB+bert/encoder/layer_1/output/LayerNorm/gammaB&bert/encoder/layer_1/output/dense/biasB(bert/encoder/layer_1/output/dense/kernelBbert/pooler/dense/biasBbert/pooler/dense/kernelB
dense/biasBdense/kernelBglobal_step
É
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
ď
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*ž
_output_shapesŤ
¨::::::::::::::::::::::::::::::::::::::::::*8
dtypes.
,2*	
N
save/Identity_1Identitysave/RestoreV2*
T0*
_output_shapes
:
g
save/AssignVariableOpAssignVariableOpbert/embeddings/LayerNorm/betasave/Identity_1*
dtype0
P
save/Identity_2Identitysave/RestoreV2:1*
T0*
_output_shapes
:
j
save/AssignVariableOp_1AssignVariableOpbert/embeddings/LayerNorm/gammasave/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:2*
T0*
_output_shapes
:
n
save/AssignVariableOp_2AssignVariableOp#bert/embeddings/position_embeddingssave/Identity_3*
dtype0
P
save/Identity_4Identitysave/RestoreV2:3*
T0*
_output_shapes
:
p
save/AssignVariableOp_3AssignVariableOp%bert/embeddings/token_type_embeddingssave/Identity_4*
dtype0
P
save/Identity_5Identitysave/RestoreV2:4*
T0*
_output_shapes
:
j
save/AssignVariableOp_4AssignVariableOpbert/embeddings/word_embeddingssave/Identity_5*
dtype0
P
save/Identity_6Identitysave/RestoreV2:5*
T0*
_output_shapes
:

save/AssignVariableOp_5AssignVariableOp4bert/encoder/layer_0/attention/output/LayerNorm/betasave/Identity_6*
dtype0
P
save/Identity_7Identitysave/RestoreV2:6*
T0*
_output_shapes
:

save/AssignVariableOp_6AssignVariableOp5bert/encoder/layer_0/attention/output/LayerNorm/gammasave/Identity_7*
dtype0
P
save/Identity_8Identitysave/RestoreV2:7*
T0*
_output_shapes
:
{
save/AssignVariableOp_7AssignVariableOp0bert/encoder/layer_0/attention/output/dense/biassave/Identity_8*
dtype0
P
save/Identity_9Identitysave/RestoreV2:8*
T0*
_output_shapes
:
}
save/AssignVariableOp_8AssignVariableOp2bert/encoder/layer_0/attention/output/dense/kernelsave/Identity_9*
dtype0
Q
save/Identity_10Identitysave/RestoreV2:9*
T0*
_output_shapes
:
x
save/AssignVariableOp_9AssignVariableOp,bert/encoder/layer_0/attention/self/key/biassave/Identity_10*
dtype0
R
save/Identity_11Identitysave/RestoreV2:10*
T0*
_output_shapes
:
{
save/AssignVariableOp_10AssignVariableOp.bert/encoder/layer_0/attention/self/key/kernelsave/Identity_11*
dtype0
R
save/Identity_12Identitysave/RestoreV2:11*
T0*
_output_shapes
:
{
save/AssignVariableOp_11AssignVariableOp.bert/encoder/layer_0/attention/self/query/biassave/Identity_12*
dtype0
R
save/Identity_13Identitysave/RestoreV2:12*
T0*
_output_shapes
:
}
save/AssignVariableOp_12AssignVariableOp0bert/encoder/layer_0/attention/self/query/kernelsave/Identity_13*
dtype0
R
save/Identity_14Identitysave/RestoreV2:13*
T0*
_output_shapes
:
{
save/AssignVariableOp_13AssignVariableOp.bert/encoder/layer_0/attention/self/value/biassave/Identity_14*
dtype0
R
save/Identity_15Identitysave/RestoreV2:14*
T0*
_output_shapes
:
}
save/AssignVariableOp_14AssignVariableOp0bert/encoder/layer_0/attention/self/value/kernelsave/Identity_15*
dtype0
R
save/Identity_16Identitysave/RestoreV2:15*
T0*
_output_shapes
:
y
save/AssignVariableOp_15AssignVariableOp,bert/encoder/layer_0/intermediate/dense/biassave/Identity_16*
dtype0
R
save/Identity_17Identitysave/RestoreV2:16*
T0*
_output_shapes
:
{
save/AssignVariableOp_16AssignVariableOp.bert/encoder/layer_0/intermediate/dense/kernelsave/Identity_17*
dtype0
R
save/Identity_18Identitysave/RestoreV2:17*
T0*
_output_shapes
:
w
save/AssignVariableOp_17AssignVariableOp*bert/encoder/layer_0/output/LayerNorm/betasave/Identity_18*
dtype0
R
save/Identity_19Identitysave/RestoreV2:18*
T0*
_output_shapes
:
x
save/AssignVariableOp_18AssignVariableOp+bert/encoder/layer_0/output/LayerNorm/gammasave/Identity_19*
dtype0
R
save/Identity_20Identitysave/RestoreV2:19*
T0*
_output_shapes
:
s
save/AssignVariableOp_19AssignVariableOp&bert/encoder/layer_0/output/dense/biassave/Identity_20*
dtype0
R
save/Identity_21Identitysave/RestoreV2:20*
T0*
_output_shapes
:
u
save/AssignVariableOp_20AssignVariableOp(bert/encoder/layer_0/output/dense/kernelsave/Identity_21*
dtype0
R
save/Identity_22Identitysave/RestoreV2:21*
T0*
_output_shapes
:

save/AssignVariableOp_21AssignVariableOp4bert/encoder/layer_1/attention/output/LayerNorm/betasave/Identity_22*
dtype0
R
save/Identity_23Identitysave/RestoreV2:22*
T0*
_output_shapes
:

save/AssignVariableOp_22AssignVariableOp5bert/encoder/layer_1/attention/output/LayerNorm/gammasave/Identity_23*
dtype0
R
save/Identity_24Identitysave/RestoreV2:23*
T0*
_output_shapes
:
}
save/AssignVariableOp_23AssignVariableOp0bert/encoder/layer_1/attention/output/dense/biassave/Identity_24*
dtype0
R
save/Identity_25Identitysave/RestoreV2:24*
T0*
_output_shapes
:

save/AssignVariableOp_24AssignVariableOp2bert/encoder/layer_1/attention/output/dense/kernelsave/Identity_25*
dtype0
R
save/Identity_26Identitysave/RestoreV2:25*
T0*
_output_shapes
:
y
save/AssignVariableOp_25AssignVariableOp,bert/encoder/layer_1/attention/self/key/biassave/Identity_26*
dtype0
R
save/Identity_27Identitysave/RestoreV2:26*
T0*
_output_shapes
:
{
save/AssignVariableOp_26AssignVariableOp.bert/encoder/layer_1/attention/self/key/kernelsave/Identity_27*
dtype0
R
save/Identity_28Identitysave/RestoreV2:27*
T0*
_output_shapes
:
{
save/AssignVariableOp_27AssignVariableOp.bert/encoder/layer_1/attention/self/query/biassave/Identity_28*
dtype0
R
save/Identity_29Identitysave/RestoreV2:28*
T0*
_output_shapes
:
}
save/AssignVariableOp_28AssignVariableOp0bert/encoder/layer_1/attention/self/query/kernelsave/Identity_29*
dtype0
R
save/Identity_30Identitysave/RestoreV2:29*
T0*
_output_shapes
:
{
save/AssignVariableOp_29AssignVariableOp.bert/encoder/layer_1/attention/self/value/biassave/Identity_30*
dtype0
R
save/Identity_31Identitysave/RestoreV2:30*
T0*
_output_shapes
:
}
save/AssignVariableOp_30AssignVariableOp0bert/encoder/layer_1/attention/self/value/kernelsave/Identity_31*
dtype0
R
save/Identity_32Identitysave/RestoreV2:31*
T0*
_output_shapes
:
y
save/AssignVariableOp_31AssignVariableOp,bert/encoder/layer_1/intermediate/dense/biassave/Identity_32*
dtype0
R
save/Identity_33Identitysave/RestoreV2:32*
T0*
_output_shapes
:
{
save/AssignVariableOp_32AssignVariableOp.bert/encoder/layer_1/intermediate/dense/kernelsave/Identity_33*
dtype0
R
save/Identity_34Identitysave/RestoreV2:33*
T0*
_output_shapes
:
w
save/AssignVariableOp_33AssignVariableOp*bert/encoder/layer_1/output/LayerNorm/betasave/Identity_34*
dtype0
R
save/Identity_35Identitysave/RestoreV2:34*
T0*
_output_shapes
:
x
save/AssignVariableOp_34AssignVariableOp+bert/encoder/layer_1/output/LayerNorm/gammasave/Identity_35*
dtype0
R
save/Identity_36Identitysave/RestoreV2:35*
T0*
_output_shapes
:
s
save/AssignVariableOp_35AssignVariableOp&bert/encoder/layer_1/output/dense/biassave/Identity_36*
dtype0
R
save/Identity_37Identitysave/RestoreV2:36*
T0*
_output_shapes
:
u
save/AssignVariableOp_36AssignVariableOp(bert/encoder/layer_1/output/dense/kernelsave/Identity_37*
dtype0
R
save/Identity_38Identitysave/RestoreV2:37*
T0*
_output_shapes
:
c
save/AssignVariableOp_37AssignVariableOpbert/pooler/dense/biassave/Identity_38*
dtype0
R
save/Identity_39Identitysave/RestoreV2:38*
T0*
_output_shapes
:
e
save/AssignVariableOp_38AssignVariableOpbert/pooler/dense/kernelsave/Identity_39*
dtype0
R
save/Identity_40Identitysave/RestoreV2:39*
T0*
_output_shapes
:
W
save/AssignVariableOp_39AssignVariableOp
dense/biassave/Identity_40*
dtype0
R
save/Identity_41Identitysave/RestoreV2:40*
T0*
_output_shapes
:
Y
save/AssignVariableOp_40AssignVariableOpdense/kernelsave/Identity_41*
dtype0
R
save/Identity_42Identitysave/RestoreV2:41*
T0	*
_output_shapes
:
X
save/AssignVariableOp_41AssignVariableOpglobal_stepsave/Identity_42*
dtype0	
ü
save/restore_shardNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_10^save/AssignVariableOp_11^save/AssignVariableOp_12^save/AssignVariableOp_13^save/AssignVariableOp_14^save/AssignVariableOp_15^save/AssignVariableOp_16^save/AssignVariableOp_17^save/AssignVariableOp_18^save/AssignVariableOp_19^save/AssignVariableOp_2^save/AssignVariableOp_20^save/AssignVariableOp_21^save/AssignVariableOp_22^save/AssignVariableOp_23^save/AssignVariableOp_24^save/AssignVariableOp_25^save/AssignVariableOp_26^save/AssignVariableOp_27^save/AssignVariableOp_28^save/AssignVariableOp_29^save/AssignVariableOp_3^save/AssignVariableOp_30^save/AssignVariableOp_31^save/AssignVariableOp_32^save/AssignVariableOp_33^save/AssignVariableOp_34^save/AssignVariableOp_35^save/AssignVariableOp_36^save/AssignVariableOp_37^save/AssignVariableOp_38^save/AssignVariableOp_39^save/AssignVariableOp_4^save/AssignVariableOp_40^save/AssignVariableOp_41^save/AssignVariableOp_5^save/AssignVariableOp_6^save/AssignVariableOp_7^save/AssignVariableOp_8^save/AssignVariableOp_9
-
save/restore_allNoOp^save/restore_shard"Ď<
save/Const:0save/Identity:0save/restore_all (5 @F8"~
global_stepom
k
global_step:0global_step/Assign!global_step/Read/ReadVariableOp:0(2global_step/Initializer/zeros:0H"
model_variablesöó
ˇ
 bert/embeddings/LayerNorm/beta:0%bert/embeddings/LayerNorm/beta/Assign4bert/embeddings/LayerNorm/beta/Read/ReadVariableOp:0(22bert/embeddings/LayerNorm/beta/Initializer/zeros:08
ş
!bert/embeddings/LayerNorm/gamma:0&bert/embeddings/LayerNorm/gamma/Assign5bert/embeddings/LayerNorm/gamma/Read/ReadVariableOp:0(22bert/embeddings/LayerNorm/gamma/Initializer/ones:08

6bert/encoder/layer_0/attention/output/LayerNorm/beta:0;bert/encoder/layer_0/attention/output/LayerNorm/beta/AssignJbert/encoder/layer_0/attention/output/LayerNorm/beta/Read/ReadVariableOp:0(2Hbert/encoder/layer_0/attention/output/LayerNorm/beta/Initializer/zeros:08

7bert/encoder/layer_0/attention/output/LayerNorm/gamma:0<bert/encoder/layer_0/attention/output/LayerNorm/gamma/AssignKbert/encoder/layer_0/attention/output/LayerNorm/gamma/Read/ReadVariableOp:0(2Hbert/encoder/layer_0/attention/output/LayerNorm/gamma/Initializer/ones:08
ç
,bert/encoder/layer_0/output/LayerNorm/beta:01bert/encoder/layer_0/output/LayerNorm/beta/Assign@bert/encoder/layer_0/output/LayerNorm/beta/Read/ReadVariableOp:0(2>bert/encoder/layer_0/output/LayerNorm/beta/Initializer/zeros:08
ę
-bert/encoder/layer_0/output/LayerNorm/gamma:02bert/encoder/layer_0/output/LayerNorm/gamma/AssignAbert/encoder/layer_0/output/LayerNorm/gamma/Read/ReadVariableOp:0(2>bert/encoder/layer_0/output/LayerNorm/gamma/Initializer/ones:08

6bert/encoder/layer_1/attention/output/LayerNorm/beta:0;bert/encoder/layer_1/attention/output/LayerNorm/beta/AssignJbert/encoder/layer_1/attention/output/LayerNorm/beta/Read/ReadVariableOp:0(2Hbert/encoder/layer_1/attention/output/LayerNorm/beta/Initializer/zeros:08

7bert/encoder/layer_1/attention/output/LayerNorm/gamma:0<bert/encoder/layer_1/attention/output/LayerNorm/gamma/AssignKbert/encoder/layer_1/attention/output/LayerNorm/gamma/Read/ReadVariableOp:0(2Hbert/encoder/layer_1/attention/output/LayerNorm/gamma/Initializer/ones:08
ç
,bert/encoder/layer_1/output/LayerNorm/beta:01bert/encoder/layer_1/output/LayerNorm/beta/Assign@bert/encoder/layer_1/output/LayerNorm/beta/Read/ReadVariableOp:0(2>bert/encoder/layer_1/output/LayerNorm/beta/Initializer/zeros:08
ę
-bert/encoder/layer_1/output/LayerNorm/gamma:02bert/encoder/layer_1/output/LayerNorm/gamma/AssignAbert/encoder/layer_1/output/LayerNorm/gamma/Read/ReadVariableOp:0(2>bert/encoder/layer_1/output/LayerNorm/gamma/Initializer/ones:08"%
saved_model_main_op


group_deps"ďK
trainable_variables×KÔK
Ć
!bert/embeddings/word_embeddings:0&bert/embeddings/word_embeddings/Assign5bert/embeddings/word_embeddings/Read/ReadVariableOp:0(2>bert/embeddings/word_embeddings/Initializer/truncated_normal:08
Ţ
'bert/embeddings/token_type_embeddings:0,bert/embeddings/token_type_embeddings/Assign;bert/embeddings/token_type_embeddings/Read/ReadVariableOp:0(2Dbert/embeddings/token_type_embeddings/Initializer/truncated_normal:08
Ö
%bert/embeddings/position_embeddings:0*bert/embeddings/position_embeddings/Assign9bert/embeddings/position_embeddings/Read/ReadVariableOp:0(2Bbert/embeddings/position_embeddings/Initializer/truncated_normal:08
ˇ
 bert/embeddings/LayerNorm/beta:0%bert/embeddings/LayerNorm/beta/Assign4bert/embeddings/LayerNorm/beta/Read/ReadVariableOp:0(22bert/embeddings/LayerNorm/beta/Initializer/zeros:08
ş
!bert/embeddings/LayerNorm/gamma:0&bert/embeddings/LayerNorm/gamma/Assign5bert/embeddings/LayerNorm/gamma/Read/ReadVariableOp:0(22bert/embeddings/LayerNorm/gamma/Initializer/ones:08

2bert/encoder/layer_0/attention/self/query/kernel:07bert/encoder/layer_0/attention/self/query/kernel/AssignFbert/encoder/layer_0/attention/self/query/kernel/Read/ReadVariableOp:0(2Obert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal:08
÷
0bert/encoder/layer_0/attention/self/query/bias:05bert/encoder/layer_0/attention/self/query/bias/AssignDbert/encoder/layer_0/attention/self/query/bias/Read/ReadVariableOp:0(2Bbert/encoder/layer_0/attention/self/query/bias/Initializer/zeros:08

0bert/encoder/layer_0/attention/self/key/kernel:05bert/encoder/layer_0/attention/self/key/kernel/AssignDbert/encoder/layer_0/attention/self/key/kernel/Read/ReadVariableOp:0(2Mbert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal:08
ď
.bert/encoder/layer_0/attention/self/key/bias:03bert/encoder/layer_0/attention/self/key/bias/AssignBbert/encoder/layer_0/attention/self/key/bias/Read/ReadVariableOp:0(2@bert/encoder/layer_0/attention/self/key/bias/Initializer/zeros:08

2bert/encoder/layer_0/attention/self/value/kernel:07bert/encoder/layer_0/attention/self/value/kernel/AssignFbert/encoder/layer_0/attention/self/value/kernel/Read/ReadVariableOp:0(2Obert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal:08
÷
0bert/encoder/layer_0/attention/self/value/bias:05bert/encoder/layer_0/attention/self/value/bias/AssignDbert/encoder/layer_0/attention/self/value/bias/Read/ReadVariableOp:0(2Bbert/encoder/layer_0/attention/self/value/bias/Initializer/zeros:08

4bert/encoder/layer_0/attention/output/dense/kernel:09bert/encoder/layer_0/attention/output/dense/kernel/AssignHbert/encoder/layer_0/attention/output/dense/kernel/Read/ReadVariableOp:0(2Qbert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal:08
˙
2bert/encoder/layer_0/attention/output/dense/bias:07bert/encoder/layer_0/attention/output/dense/bias/AssignFbert/encoder/layer_0/attention/output/dense/bias/Read/ReadVariableOp:0(2Dbert/encoder/layer_0/attention/output/dense/bias/Initializer/zeros:08

6bert/encoder/layer_0/attention/output/LayerNorm/beta:0;bert/encoder/layer_0/attention/output/LayerNorm/beta/AssignJbert/encoder/layer_0/attention/output/LayerNorm/beta/Read/ReadVariableOp:0(2Hbert/encoder/layer_0/attention/output/LayerNorm/beta/Initializer/zeros:08

7bert/encoder/layer_0/attention/output/LayerNorm/gamma:0<bert/encoder/layer_0/attention/output/LayerNorm/gamma/AssignKbert/encoder/layer_0/attention/output/LayerNorm/gamma/Read/ReadVariableOp:0(2Hbert/encoder/layer_0/attention/output/LayerNorm/gamma/Initializer/ones:08

0bert/encoder/layer_0/intermediate/dense/kernel:05bert/encoder/layer_0/intermediate/dense/kernel/AssignDbert/encoder/layer_0/intermediate/dense/kernel/Read/ReadVariableOp:0(2Mbert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal:08
ď
.bert/encoder/layer_0/intermediate/dense/bias:03bert/encoder/layer_0/intermediate/dense/bias/AssignBbert/encoder/layer_0/intermediate/dense/bias/Read/ReadVariableOp:0(2@bert/encoder/layer_0/intermediate/dense/bias/Initializer/zeros:08
ę
*bert/encoder/layer_0/output/dense/kernel:0/bert/encoder/layer_0/output/dense/kernel/Assign>bert/encoder/layer_0/output/dense/kernel/Read/ReadVariableOp:0(2Gbert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal:08
×
(bert/encoder/layer_0/output/dense/bias:0-bert/encoder/layer_0/output/dense/bias/Assign<bert/encoder/layer_0/output/dense/bias/Read/ReadVariableOp:0(2:bert/encoder/layer_0/output/dense/bias/Initializer/zeros:08
ç
,bert/encoder/layer_0/output/LayerNorm/beta:01bert/encoder/layer_0/output/LayerNorm/beta/Assign@bert/encoder/layer_0/output/LayerNorm/beta/Read/ReadVariableOp:0(2>bert/encoder/layer_0/output/LayerNorm/beta/Initializer/zeros:08
ę
-bert/encoder/layer_0/output/LayerNorm/gamma:02bert/encoder/layer_0/output/LayerNorm/gamma/AssignAbert/encoder/layer_0/output/LayerNorm/gamma/Read/ReadVariableOp:0(2>bert/encoder/layer_0/output/LayerNorm/gamma/Initializer/ones:08

2bert/encoder/layer_1/attention/self/query/kernel:07bert/encoder/layer_1/attention/self/query/kernel/AssignFbert/encoder/layer_1/attention/self/query/kernel/Read/ReadVariableOp:0(2Obert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal:08
÷
0bert/encoder/layer_1/attention/self/query/bias:05bert/encoder/layer_1/attention/self/query/bias/AssignDbert/encoder/layer_1/attention/self/query/bias/Read/ReadVariableOp:0(2Bbert/encoder/layer_1/attention/self/query/bias/Initializer/zeros:08

0bert/encoder/layer_1/attention/self/key/kernel:05bert/encoder/layer_1/attention/self/key/kernel/AssignDbert/encoder/layer_1/attention/self/key/kernel/Read/ReadVariableOp:0(2Mbert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal:08
ď
.bert/encoder/layer_1/attention/self/key/bias:03bert/encoder/layer_1/attention/self/key/bias/AssignBbert/encoder/layer_1/attention/self/key/bias/Read/ReadVariableOp:0(2@bert/encoder/layer_1/attention/self/key/bias/Initializer/zeros:08

2bert/encoder/layer_1/attention/self/value/kernel:07bert/encoder/layer_1/attention/self/value/kernel/AssignFbert/encoder/layer_1/attention/self/value/kernel/Read/ReadVariableOp:0(2Obert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal:08
÷
0bert/encoder/layer_1/attention/self/value/bias:05bert/encoder/layer_1/attention/self/value/bias/AssignDbert/encoder/layer_1/attention/self/value/bias/Read/ReadVariableOp:0(2Bbert/encoder/layer_1/attention/self/value/bias/Initializer/zeros:08

4bert/encoder/layer_1/attention/output/dense/kernel:09bert/encoder/layer_1/attention/output/dense/kernel/AssignHbert/encoder/layer_1/attention/output/dense/kernel/Read/ReadVariableOp:0(2Qbert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal:08
˙
2bert/encoder/layer_1/attention/output/dense/bias:07bert/encoder/layer_1/attention/output/dense/bias/AssignFbert/encoder/layer_1/attention/output/dense/bias/Read/ReadVariableOp:0(2Dbert/encoder/layer_1/attention/output/dense/bias/Initializer/zeros:08

6bert/encoder/layer_1/attention/output/LayerNorm/beta:0;bert/encoder/layer_1/attention/output/LayerNorm/beta/AssignJbert/encoder/layer_1/attention/output/LayerNorm/beta/Read/ReadVariableOp:0(2Hbert/encoder/layer_1/attention/output/LayerNorm/beta/Initializer/zeros:08

7bert/encoder/layer_1/attention/output/LayerNorm/gamma:0<bert/encoder/layer_1/attention/output/LayerNorm/gamma/AssignKbert/encoder/layer_1/attention/output/LayerNorm/gamma/Read/ReadVariableOp:0(2Hbert/encoder/layer_1/attention/output/LayerNorm/gamma/Initializer/ones:08

0bert/encoder/layer_1/intermediate/dense/kernel:05bert/encoder/layer_1/intermediate/dense/kernel/AssignDbert/encoder/layer_1/intermediate/dense/kernel/Read/ReadVariableOp:0(2Mbert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal:08
ď
.bert/encoder/layer_1/intermediate/dense/bias:03bert/encoder/layer_1/intermediate/dense/bias/AssignBbert/encoder/layer_1/intermediate/dense/bias/Read/ReadVariableOp:0(2@bert/encoder/layer_1/intermediate/dense/bias/Initializer/zeros:08
ę
*bert/encoder/layer_1/output/dense/kernel:0/bert/encoder/layer_1/output/dense/kernel/Assign>bert/encoder/layer_1/output/dense/kernel/Read/ReadVariableOp:0(2Gbert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal:08
×
(bert/encoder/layer_1/output/dense/bias:0-bert/encoder/layer_1/output/dense/bias/Assign<bert/encoder/layer_1/output/dense/bias/Read/ReadVariableOp:0(2:bert/encoder/layer_1/output/dense/bias/Initializer/zeros:08
ç
,bert/encoder/layer_1/output/LayerNorm/beta:01bert/encoder/layer_1/output/LayerNorm/beta/Assign@bert/encoder/layer_1/output/LayerNorm/beta/Read/ReadVariableOp:0(2>bert/encoder/layer_1/output/LayerNorm/beta/Initializer/zeros:08
ę
-bert/encoder/layer_1/output/LayerNorm/gamma:02bert/encoder/layer_1/output/LayerNorm/gamma/AssignAbert/encoder/layer_1/output/LayerNorm/gamma/Read/ReadVariableOp:0(2>bert/encoder/layer_1/output/LayerNorm/gamma/Initializer/ones:08
Ş
bert/pooler/dense/kernel:0bert/pooler/dense/kernel/Assign.bert/pooler/dense/kernel/Read/ReadVariableOp:0(27bert/pooler/dense/kernel/Initializer/truncated_normal:08

bert/pooler/dense/bias:0bert/pooler/dense/bias/Assign,bert/pooler/dense/bias/Read/ReadVariableOp:0(2*bert/pooler/dense/bias/Initializer/zeros:08
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/Const:08"ŇL
	variablesÄLÁL
k
global_step:0global_step/Assign!global_step/Read/ReadVariableOp:0(2global_step/Initializer/zeros:0H
Ć
!bert/embeddings/word_embeddings:0&bert/embeddings/word_embeddings/Assign5bert/embeddings/word_embeddings/Read/ReadVariableOp:0(2>bert/embeddings/word_embeddings/Initializer/truncated_normal:08
Ţ
'bert/embeddings/token_type_embeddings:0,bert/embeddings/token_type_embeddings/Assign;bert/embeddings/token_type_embeddings/Read/ReadVariableOp:0(2Dbert/embeddings/token_type_embeddings/Initializer/truncated_normal:08
Ö
%bert/embeddings/position_embeddings:0*bert/embeddings/position_embeddings/Assign9bert/embeddings/position_embeddings/Read/ReadVariableOp:0(2Bbert/embeddings/position_embeddings/Initializer/truncated_normal:08
ˇ
 bert/embeddings/LayerNorm/beta:0%bert/embeddings/LayerNorm/beta/Assign4bert/embeddings/LayerNorm/beta/Read/ReadVariableOp:0(22bert/embeddings/LayerNorm/beta/Initializer/zeros:08
ş
!bert/embeddings/LayerNorm/gamma:0&bert/embeddings/LayerNorm/gamma/Assign5bert/embeddings/LayerNorm/gamma/Read/ReadVariableOp:0(22bert/embeddings/LayerNorm/gamma/Initializer/ones:08

2bert/encoder/layer_0/attention/self/query/kernel:07bert/encoder/layer_0/attention/self/query/kernel/AssignFbert/encoder/layer_0/attention/self/query/kernel/Read/ReadVariableOp:0(2Obert/encoder/layer_0/attention/self/query/kernel/Initializer/truncated_normal:08
÷
0bert/encoder/layer_0/attention/self/query/bias:05bert/encoder/layer_0/attention/self/query/bias/AssignDbert/encoder/layer_0/attention/self/query/bias/Read/ReadVariableOp:0(2Bbert/encoder/layer_0/attention/self/query/bias/Initializer/zeros:08

0bert/encoder/layer_0/attention/self/key/kernel:05bert/encoder/layer_0/attention/self/key/kernel/AssignDbert/encoder/layer_0/attention/self/key/kernel/Read/ReadVariableOp:0(2Mbert/encoder/layer_0/attention/self/key/kernel/Initializer/truncated_normal:08
ď
.bert/encoder/layer_0/attention/self/key/bias:03bert/encoder/layer_0/attention/self/key/bias/AssignBbert/encoder/layer_0/attention/self/key/bias/Read/ReadVariableOp:0(2@bert/encoder/layer_0/attention/self/key/bias/Initializer/zeros:08

2bert/encoder/layer_0/attention/self/value/kernel:07bert/encoder/layer_0/attention/self/value/kernel/AssignFbert/encoder/layer_0/attention/self/value/kernel/Read/ReadVariableOp:0(2Obert/encoder/layer_0/attention/self/value/kernel/Initializer/truncated_normal:08
÷
0bert/encoder/layer_0/attention/self/value/bias:05bert/encoder/layer_0/attention/self/value/bias/AssignDbert/encoder/layer_0/attention/self/value/bias/Read/ReadVariableOp:0(2Bbert/encoder/layer_0/attention/self/value/bias/Initializer/zeros:08

4bert/encoder/layer_0/attention/output/dense/kernel:09bert/encoder/layer_0/attention/output/dense/kernel/AssignHbert/encoder/layer_0/attention/output/dense/kernel/Read/ReadVariableOp:0(2Qbert/encoder/layer_0/attention/output/dense/kernel/Initializer/truncated_normal:08
˙
2bert/encoder/layer_0/attention/output/dense/bias:07bert/encoder/layer_0/attention/output/dense/bias/AssignFbert/encoder/layer_0/attention/output/dense/bias/Read/ReadVariableOp:0(2Dbert/encoder/layer_0/attention/output/dense/bias/Initializer/zeros:08

6bert/encoder/layer_0/attention/output/LayerNorm/beta:0;bert/encoder/layer_0/attention/output/LayerNorm/beta/AssignJbert/encoder/layer_0/attention/output/LayerNorm/beta/Read/ReadVariableOp:0(2Hbert/encoder/layer_0/attention/output/LayerNorm/beta/Initializer/zeros:08

7bert/encoder/layer_0/attention/output/LayerNorm/gamma:0<bert/encoder/layer_0/attention/output/LayerNorm/gamma/AssignKbert/encoder/layer_0/attention/output/LayerNorm/gamma/Read/ReadVariableOp:0(2Hbert/encoder/layer_0/attention/output/LayerNorm/gamma/Initializer/ones:08

0bert/encoder/layer_0/intermediate/dense/kernel:05bert/encoder/layer_0/intermediate/dense/kernel/AssignDbert/encoder/layer_0/intermediate/dense/kernel/Read/ReadVariableOp:0(2Mbert/encoder/layer_0/intermediate/dense/kernel/Initializer/truncated_normal:08
ď
.bert/encoder/layer_0/intermediate/dense/bias:03bert/encoder/layer_0/intermediate/dense/bias/AssignBbert/encoder/layer_0/intermediate/dense/bias/Read/ReadVariableOp:0(2@bert/encoder/layer_0/intermediate/dense/bias/Initializer/zeros:08
ę
*bert/encoder/layer_0/output/dense/kernel:0/bert/encoder/layer_0/output/dense/kernel/Assign>bert/encoder/layer_0/output/dense/kernel/Read/ReadVariableOp:0(2Gbert/encoder/layer_0/output/dense/kernel/Initializer/truncated_normal:08
×
(bert/encoder/layer_0/output/dense/bias:0-bert/encoder/layer_0/output/dense/bias/Assign<bert/encoder/layer_0/output/dense/bias/Read/ReadVariableOp:0(2:bert/encoder/layer_0/output/dense/bias/Initializer/zeros:08
ç
,bert/encoder/layer_0/output/LayerNorm/beta:01bert/encoder/layer_0/output/LayerNorm/beta/Assign@bert/encoder/layer_0/output/LayerNorm/beta/Read/ReadVariableOp:0(2>bert/encoder/layer_0/output/LayerNorm/beta/Initializer/zeros:08
ę
-bert/encoder/layer_0/output/LayerNorm/gamma:02bert/encoder/layer_0/output/LayerNorm/gamma/AssignAbert/encoder/layer_0/output/LayerNorm/gamma/Read/ReadVariableOp:0(2>bert/encoder/layer_0/output/LayerNorm/gamma/Initializer/ones:08

2bert/encoder/layer_1/attention/self/query/kernel:07bert/encoder/layer_1/attention/self/query/kernel/AssignFbert/encoder/layer_1/attention/self/query/kernel/Read/ReadVariableOp:0(2Obert/encoder/layer_1/attention/self/query/kernel/Initializer/truncated_normal:08
÷
0bert/encoder/layer_1/attention/self/query/bias:05bert/encoder/layer_1/attention/self/query/bias/AssignDbert/encoder/layer_1/attention/self/query/bias/Read/ReadVariableOp:0(2Bbert/encoder/layer_1/attention/self/query/bias/Initializer/zeros:08

0bert/encoder/layer_1/attention/self/key/kernel:05bert/encoder/layer_1/attention/self/key/kernel/AssignDbert/encoder/layer_1/attention/self/key/kernel/Read/ReadVariableOp:0(2Mbert/encoder/layer_1/attention/self/key/kernel/Initializer/truncated_normal:08
ď
.bert/encoder/layer_1/attention/self/key/bias:03bert/encoder/layer_1/attention/self/key/bias/AssignBbert/encoder/layer_1/attention/self/key/bias/Read/ReadVariableOp:0(2@bert/encoder/layer_1/attention/self/key/bias/Initializer/zeros:08

2bert/encoder/layer_1/attention/self/value/kernel:07bert/encoder/layer_1/attention/self/value/kernel/AssignFbert/encoder/layer_1/attention/self/value/kernel/Read/ReadVariableOp:0(2Obert/encoder/layer_1/attention/self/value/kernel/Initializer/truncated_normal:08
÷
0bert/encoder/layer_1/attention/self/value/bias:05bert/encoder/layer_1/attention/self/value/bias/AssignDbert/encoder/layer_1/attention/self/value/bias/Read/ReadVariableOp:0(2Bbert/encoder/layer_1/attention/self/value/bias/Initializer/zeros:08

4bert/encoder/layer_1/attention/output/dense/kernel:09bert/encoder/layer_1/attention/output/dense/kernel/AssignHbert/encoder/layer_1/attention/output/dense/kernel/Read/ReadVariableOp:0(2Qbert/encoder/layer_1/attention/output/dense/kernel/Initializer/truncated_normal:08
˙
2bert/encoder/layer_1/attention/output/dense/bias:07bert/encoder/layer_1/attention/output/dense/bias/AssignFbert/encoder/layer_1/attention/output/dense/bias/Read/ReadVariableOp:0(2Dbert/encoder/layer_1/attention/output/dense/bias/Initializer/zeros:08

6bert/encoder/layer_1/attention/output/LayerNorm/beta:0;bert/encoder/layer_1/attention/output/LayerNorm/beta/AssignJbert/encoder/layer_1/attention/output/LayerNorm/beta/Read/ReadVariableOp:0(2Hbert/encoder/layer_1/attention/output/LayerNorm/beta/Initializer/zeros:08

7bert/encoder/layer_1/attention/output/LayerNorm/gamma:0<bert/encoder/layer_1/attention/output/LayerNorm/gamma/AssignKbert/encoder/layer_1/attention/output/LayerNorm/gamma/Read/ReadVariableOp:0(2Hbert/encoder/layer_1/attention/output/LayerNorm/gamma/Initializer/ones:08

0bert/encoder/layer_1/intermediate/dense/kernel:05bert/encoder/layer_1/intermediate/dense/kernel/AssignDbert/encoder/layer_1/intermediate/dense/kernel/Read/ReadVariableOp:0(2Mbert/encoder/layer_1/intermediate/dense/kernel/Initializer/truncated_normal:08
ď
.bert/encoder/layer_1/intermediate/dense/bias:03bert/encoder/layer_1/intermediate/dense/bias/AssignBbert/encoder/layer_1/intermediate/dense/bias/Read/ReadVariableOp:0(2@bert/encoder/layer_1/intermediate/dense/bias/Initializer/zeros:08
ę
*bert/encoder/layer_1/output/dense/kernel:0/bert/encoder/layer_1/output/dense/kernel/Assign>bert/encoder/layer_1/output/dense/kernel/Read/ReadVariableOp:0(2Gbert/encoder/layer_1/output/dense/kernel/Initializer/truncated_normal:08
×
(bert/encoder/layer_1/output/dense/bias:0-bert/encoder/layer_1/output/dense/bias/Assign<bert/encoder/layer_1/output/dense/bias/Read/ReadVariableOp:0(2:bert/encoder/layer_1/output/dense/bias/Initializer/zeros:08
ç
,bert/encoder/layer_1/output/LayerNorm/beta:01bert/encoder/layer_1/output/LayerNorm/beta/Assign@bert/encoder/layer_1/output/LayerNorm/beta/Read/ReadVariableOp:0(2>bert/encoder/layer_1/output/LayerNorm/beta/Initializer/zeros:08
ę
-bert/encoder/layer_1/output/LayerNorm/gamma:02bert/encoder/layer_1/output/LayerNorm/gamma/AssignAbert/encoder/layer_1/output/LayerNorm/gamma/Read/ReadVariableOp:0(2>bert/encoder/layer_1/output/LayerNorm/gamma/Initializer/ones:08
Ş
bert/pooler/dense/kernel:0bert/pooler/dense/kernel/Assign.bert/pooler/dense/kernel/Read/ReadVariableOp:0(27bert/pooler/dense/kernel/Initializer/truncated_normal:08

bert/pooler/dense/bias:0bert/pooler/dense/bias/Assign,bert/pooler/dense/bias/Read/ReadVariableOp:0(2*bert/pooler/dense/bias/Initializer/zeros:08
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/Const:08*
serving_defaultě
2
	input_ids%
Placeholder:0	˙˙˙˙˙˙˙˙˙
5

input_mask'
Placeholder_1:0	˙˙˙˙˙˙˙˙˙
6
segment_ids'
Placeholder_2:0	˙˙˙˙˙˙˙˙˙+
predictions
	Squeeze:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict