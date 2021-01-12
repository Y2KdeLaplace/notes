# notes_of_python

## load
import scipy.io as scio
dataFile = 'E://data.mat'
data = scio.loadmat(dataFile)
注意，读取出来的data是字典格式，可以通过函数type(data)查看。
print type(data)
结果显示
<type 'dict'>

## save
将这里的data['A']矩阵重新保存到一个新的文件dataNew.mat中：
dataNew = 'E://dataNew.mat'
scio.savemat(dataNew, {'A':data['A']})
注意：是以字典的形式保存。

## pip 最常用命令
显示版本和路径

pip --version
获取帮助

pip --help
升级 pip

pip install -U pip
如果这个升级命令出现问题 ，可以使用以下命令：

sudo easy_install --upgrade pip
安装包

pip install SomePackage              # 最新版本
pip install SomePackage==1.0.4       # 指定版本
pip install 'SomePackage>=1.0.4'     # 最小版本
比如我要安装 Django。用以下的一条命令就可以，方便快捷。

pip install Django==1.7
升级包

pip install --upgrade SomePackage
升级指定的包，通过使用==, >=, <=, >, < 来指定一个版本号。

卸载包

pip uninstall SomePackage
搜索包

pip search SomePackage
显示安装包信息

pip show 
查看指定包的详细信息

pip show -f SomePackage
列出已安装的包

pip list
查看可升级的包

pip list -o
pip 升级
Linux 或 macOS

pip install --upgrade pip    # python2.x
pip3 install --upgrade pip   # python3.x
Windows 平台升级：
python -m pip install -U pip   # python2.x
python -m pip3 install -U pip    # python3.x

## 与 Matlab 比较
https://www.numpy.org.cn/user/numpy_for_matlab_users.html#%E5%A4%87%E6%B3%A8
#介绍
MATLAB®和NumPy / SciPy有很多共同之处。但是有很多不同之处。创建NumPy和SciPy是为了用Python最自然的方式进行数值和科学计算，而不是MATLAB®克隆。本页面旨在收集有关差异的智慧，主要是为了帮助熟练的MATLAB®用户成为熟练的NumPy和SciPy用户。

#一些关键的差异
MATLAB	NumPy
在MATLAB®中，基本数据类型是双精度浮点数的多维数组。大多数表达式采用这样的数组并返回这样的数 对这些数组的2-D实例的操作被设计成或多或少地像线性代数中的矩阵运算。	在NumPy中，基本类型是多维的array。包括2D在内的所有维度中对这些数组的操作是逐元素操作。人们需要使用线性代数的特定函数（尽管对于矩阵乘法，可以@在python 3.5及更高版本中使用运算符）。
MATLAB®使用基于1（一）的索引。使用（1）找到序列的初始元素。 请参阅备注	Python使用基于0（零）的索引。使用[0]找到序列的初始元素。
MATLAB®的脚本语言是为执行线性代数而创建的。基本矩阵操作的语法很好而且干净，但是用于添加GUI和制作完整应用程序的API或多或少都是事后的想法。	NumPy基于Python，它从一开始就被设计成一种优秀的通用编程语言。虽然Matlab的一些数组操作的语法比NumPy更紧凑，但NumPy（由于是Python的附加组件）可以做许多Matlab不能做的事情，例如正确处理矩阵堆栈。
在MATLAB®中，数组具有按值传递的语义，并具有惰性写入时复制方案，以防止在实际需要之前实际创建副本。切片操作复制数组的一部分。	在NumPy数组中有传递引用语义。切片操作是对数组的视图。
#'array'或'matrix'？我应该使用哪个？
从历史上看，NumPy提供了一种特殊的矩阵类型 np.matrix ，它是ndarray的子​​类，它使二进制运算成为线性代数运算。您可能会在某些现有代码中看到它而不是 np.array 。那么，使用哪一个？

#简答
使用数组。

它们是numpy的标准矢量/矩阵/张量类型。许多numpy函数返回数组，而不是矩阵。
元素操作和线性代数操作之间有明显的区别。
如果您愿意，可以使用标准向量或行/列向量。
在Python 3.5之前，使用数组类型的唯一缺点是你必须使用dot而不是*乘法（减少）两个张量（标量乘积，矩阵向量乘法等）。从Python 3.5开始，您可以使用矩阵乘法@运算符。

鉴于上述情况，我们打算matrix最终弃用。

#长答案
NumPy包含array类和matrix类。所述 array类旨在是对许多种数值计算的通用n维数组中，而matrix意在具体促进线性代数计算。在实践中，两者之间只有少数关键差异。

运算符*和@函数dot()，以及multiply()：
对于数组，*表示逐元素乘法，而 @ 表示矩阵乘法; 它们具有相关的函数 multiply() 和 dot() 。（在python 3.5之前，@ 不存在，并且必须使用dot() 进行矩阵乘法）。
对于矩阵，* 表示矩阵乘法，对于逐元素乘法，必须使用 multiply() 函数。
矢量处理（一维数组）
对于数组，向量形状1xN，Nx1和N都是不同的东西。 像 A[:, 1] 这样的操作返回形状N的一维数组，而不是形状Nx1的二维数组。 在一维数组上转置什么都不做。
对于矩阵，一维数组总是被上变频为1xN或Nx1矩阵（行或列向量）。A[:, 1] 返回形状为Nx1的二维矩阵。
处理更高维数组（ndim> 2）
数组对象的维数可以 > 2 ;
矩阵对象总是有两个维度。
便利属性
array 有一个.T属性，它返回数据的转置。
matrix 还有.H，.I和.A属性，分别返回共轭转置，反转和asarray()矩阵。
便利构造函数
该array构造采用（嵌套）的Python序列初始化。如：array([[1,2,3],[4,5,6]])。
该matrix构造还需要一个方便的字符串初始化。如：matrix("[1 2 3; 4 5 6]")。
使用两者有利有弊：

array
:) 元素乘法很容易：A*B。
:( 你必须记住，矩阵乘法有自己的运算符@。
:) 可以将一维数组视为行向量或列向量。 A @ v 将 v 视为列向量，而 v @ A 将 v 视为行向量。这可以节省您键入许多转置。
:) array 是“默认”NumPy类型，因此它获得的测试最多，并且是使用NumPy的第三方代码最有可能返回的类型。
:) 非常擅长处理任何维度的数据。
:) 如果你熟悉那么语义学更接近张量代数。
:) 所有 操作（*，/，+，-等）逐元素。
:( 稀疏矩阵scipy.sparse不与数组交互。
matrix
:\\ 行为更像MATLAB®矩阵。
<:( 最大二维。要保存您需要的三维数据，array或者可能是Python列表matrix。
<:( 最小二维。你不能有载体。它们必须作为单列或单行矩阵进行转换。
<:( 由于array是NumPy中的默认值，因此array即使您将它们matrix作为参数给出，某些函数也可能返回。这不应该发生在NumPy函数中（如果它确实是一个错误），但基于NumPy的第三方代码可能不像NumPy那样遵守类型保存。
:) A*B是矩阵乘法，所以它看起来就像你在线性代数中写的那样（对于Python> = 3.5普通数组与@运算符具有相同的便利性）。
<:( 元素乘法需要调用函数， multiply(A,B)。
<:( 运算符重载的使用有点不合逻辑：* 不能按元素操作，但 / 确实如此。
与之互动scipy.sparse有点清洁。
因此，使用 数组（array） 要明智得多。事实上，我们打算最终废除 矩阵（matrix）。

#MATLAB 和 NumPy粗略的功能对应表
下表给出了一些常见MATLAB®表达式的粗略等价物。这些不是确切的等价物，而应该作为提示让你朝着正确的方向前进。有关更多详细信息，请阅读NumPy函数的内置文档。

在下表中，假设您已在Python中执行以下命令：

from numpy import *
import scipy.linalg
另外如果下表中的注释这一列的内容是和 “matrix” 有关的话，那么参数一定是二维的形式。

#一般功能的对应表
MATLAB	NumPy	注释
help func	info(func)或者help(func)或func?（在IPython的）	获得函数func的帮助
which func	请参阅备注	找出func定义的位置
type func	source(func)或者func??（在Ipython中）	func的打印源（如果不是本机函数）
a && b	a and b	短路逻辑AND运算符（Python本机运算符）; 只有标量参数
a		b
1*i，1*j， 1i，1j	1j	复数
eps	np.spacing(1)	1与最近的浮点数之间的距离。
ode45	scipy.integrate.solve_ivp(f)	将ODE与Runge-Kutta 4,5集成
ode15s	scipy.integrate.solve_ivp(f, method='BDF')	将ODE与BDF方法集成
#线性代数功能对应表
MATLAB	NumPy	注释
ndims(a)	ndim(a) 要么 a.ndim	获取数组的维数
numel(a)	size(a) 要么 a.size	获取数组的元素数
size(a)	shape(a) 要么 a.shape	得到矩阵的“大小”
size(a,n)	a.shape[n-1]	获取数组第n维元素的数量a。（请注意，MATLAB®使用基于1的索引，而Python使用基于0的索引，请参阅备注）
[ 1 2 3; 4 5 6 ]	array([[1.,2.,3.], [4.,5.,6.]])	2x3矩阵文字
[ a b; c d ]	block([[a,b], [c,d]])	从块构造一个矩阵a，b，c，和d
a(end)	a[-1]	访问1xn矩阵中的最后一个元素 a
a(2,5)	a[1,4]	访问第二行，第五列中的元素
a(2,:)	a[1] 或者 a[1,:]	a的第二行
a(1:5,:)	a[0:5]或a[:5]或a[0:5,:]	前五行 a
a(end-4:end,:)	a[-5:]	a的最后五行
a(1:3,5:9)	a[0:3][:,4:9]	a的第一至第三行与第五至第九列交叉的元素。这提供了只读访问权限。
a([2,4,5],[1,3])	a[ix_([1,3,4],[0,2])]	第2,4,5行与第1,3列交叉的元素。这允许修改矩阵，并且不需要常规切片。
a(3:2:21,:)	a[ 2:21:2,:]	返回a的第3行与第21行之间每隔一行的行，即第3行与第21行之间的a的奇数行
a(1:2:end,:)	a[ ::2,:]	返回a的奇数行
a(end: -1:1,:) 要么 flipud(a)	a[ ::-1,:]	以相反的顺序排列的a的行
a([1:end 1],: )	a[r_[:len(a),0]]	a 的第一行添加到a的末尾行的副本
a.'	a.transpose() 要么 a.T	转置 a
a'	a.conj().transpose() 要么 a.conj().T	共轭转置 a
a * b	a @ b	矩阵乘法
a .* b	a * b	元素乘法
a./b	a/b	元素除法
a.^3	a**3	元素取幂
(a>0.5)	(a>0.5)	其i，jth元素为（a_ij> 0.5）的矩阵。Matlab结果是一个0和1的数组。NumPy结果是布尔值的数组False和True。
find(a>0.5)	nonzero(a>0.5)	找到a中所有大于0.5的元素的线性位置
a(:,find(v>0.5))	a[:,nonzero(v>0.5)[0]]	提取a中向量v> 0.5的对应的列
a(:,find(v>0.5))	a[:,v.T>0.5]	提取a中向量v> 0.5的对应的列
a(a<0.5)=0	a[a<0.5]=0	a中小于0.5的元素赋值为0
a .* (a>0.5)	a * (a>0.5)	返回一个数组，若a中对应位置元素大于0.5，取该元素的值；若a中对应元素<=0.5，取值0
a(: ) = 3	a[:] = 3	将所有值设置为相同的标量值
y=x	y = x.copy()	numpy通过引用分配
y=x(2,:)	y = x[1,:].copy()	numpy切片是参考
y=x(: )	y = x.flatten()	将数组转换为向量（请注意，这会强制复制）
1:10	arange(1.,11.)或r_[1.:11.]或 r_[1:10:10j]	创建一个增加的向量，步长为默认值1（参见备注）
0:9	arange(10.)或 r_[:10.]或 r_[:9:10j]	创建一个增加的向量，步长为默认值1（参见注释范围）
[1:10]'	arange(1.,11.)[:, newaxis]	创建列向量
zeros(3,4)	zeros((3,4))	3x4二维数组，充满64位浮点零
zeros(3,4,5)	zeros((3,4,5))	3x4x5三维数组，全部为64位浮点零
ones(3,4)	ones((3,4))	3x4二维数组，充满64位浮点数
eye(3)	eye(3)	3x3单位矩阵
diag(a)	diag(a)	返回a的对角元素
diag(a,0)	diag(a,0)	方形对角矩阵，其非零值是元素 a
rand(3,4)	random.rand(3,4) 要么 random.random_sample((3, 4))	随机3x4矩阵
linspace(1,3,4)	linspace(1,3,4)	4个等间距的样本，介于1和3之间
[x,y]=meshgrid(0:8,0:5)	mgrid[0:9.,0:6.] 要么 meshgrid(r_[0:9.],r_[0:6.]	两个2D数组：一个是x值，另一个是y值
ogrid[0:9.,0:6.] 要么 ix_(r_[0:9.],r_[0:6.]	在网格上评估函数的最佳方法	
[x,y]=meshgrid([1,2,4],[2,4,5])	meshgrid([1,2,4],[2,4,5])	
ix_([1,2,4],[2,4,5])	在网格上评估函数的最佳方法	
repmat(a, m, n)	tile(a, (m, n))	用n份副本创建m a
[a b]	concatenate((a,b),1)或者hstack((a,b))或 column_stack((a,b))或c_[a,b]	连接a和的列b
[a; b]	concatenate((a,b))或vstack((a,b))或r_[a,b]	连接a和的行b
max(max(a))	a.max()	最大元素a（对于matlab，ndims（a）<= 2）
max(a)	a.max(0)	每列矩阵的最大元素 a
max(a,[],2)	a.max(1)	每行矩阵的最大元素 a
max(a,b)	maximum(a, b)	比较a和b逐个元素，并返回每对中的最大值
norm(v)	sqrt(v @ v) 要么 np.linalg.norm(v)	L2矢量的规范 v
a & b	logical_and(a,b)	逐个元素AND运算符（NumPy ufunc）请参阅备注LOGICOPS
a	b	logical_or(a,b)
bitand(a,b)	a & b	按位AND运算符（Python native和NumPy ufunc）
bitor(a,b)	a	b
inv(a)	linalg.inv(a)	方阵的逆 a
pinv(a)	linalg.pinv(a)	矩阵的伪逆 a
rank(a)	linalg.matrix_rank(a)	二维数组/矩阵的矩阵秩 a
a\b	linalg.solve(a,b)如果a是正方形; linalg.lstsq(a,b) 除此以外	ax = b的解为x
b/a	解决aT xT = bT	xa = b的解为x
[U,S,V]=svd(a)	U, S, Vh = linalg.svd(a), V = Vh.T	奇异值分解 a
chol(a)	linalg.cholesky(a).T	矩阵的cholesky分解（chol(a)在matlab中返回一个上三角矩阵，但linalg.cholesky(a)返回一个下三角矩阵）
[V,D]=eig(a)	D,V = linalg.eig(a)	特征值和特征向量 a
[V,D]=eig(a,b)	D,V = scipy.linalg.eig(a,b)	特征值和特征向量a，b
[V,D]=eigs(a,k)		找到k最大的特征值和特征向量a
[Q,R,P]=qr(a,0)	Q,R = scipy.linalg.qr(a)	QR分解
[L,U,P]=lu(a)	L,U = scipy.linalg.lu(a) 要么 LU,P=scipy.linalg.lu_factor(a)	LU分解（注：P（Matlab）==转置（P（numpy）））
conjgrad	scipy.sparse.linalg.cg	共轭渐变求解器
fft(a)	fft(a)	傅立叶变换 a
ifft(a)	ifft(a)	逆傅立叶变换 a
sort(a)	sort(a) 要么 a.sort()	对矩阵进行排序
[b,I] = sortrows(a,i)	I = argsort(a[:,i]), b=a[I,:]	对矩阵的行进行排序
regress(y,X)	linalg.lstsq(X,y)	多线性回归
decimate(x, q)	scipy.signal.resample(x, len(x)/q)	采用低通滤波的下采样
unique(a)	unique(a)	
squeeze(a)	a.squeeze()	
#备注
子矩阵：使用该ix_命令可以使用索引列表完成对子矩阵的分配。例如，对于2D数组a，可能会做：ind=[1,3]; a[np.ix_(ind,ind)]+=100。

帮助：有MATLAB的没有直接等价which的命令，但命令help和source通常会列出其中函数所在的文件名。Python还有一个inspect模块（do import inspect），它提供了一个getfile经常工作的模块。

索引：MATLAB®使用一个基于索引，因此序列的初始元素具有索引1.Python使用基于零的索引，因此序列的初始元素具有索引0.出现混淆和火焰，因为每个元素都有优点和缺点。一种基于索引的方法与常见的人类语言使用一致，其中序列的“第一”元素具有索引1.基于零的索引简化了索引。另见prof.dr的文本。Edsger W. Dijkstra。

范围：在MATLAB®中，0:5可以用作范围文字和“切片”索引（括号内）; 然而，在Python，构建体等0:5可以 仅 被用作切片指数（方括号内）。因此，r_创建了一些有点古怪的对象，以使numpy具有类似的简洁范围构造机制。请注意，r_它不像函数或构造函数那样 被调用，而是 使用方括号进行 索引 ，这允许在参数中使用Python的切片语法。

逻辑运算：＆或| 在NumPy中是按位AND / OR，而在Matlab＆和|中 是逻辑AND / OR。任何具有重要编程经验的人都应该清楚这种差异。这两者似乎工作原理相同，但存在重要差异。如果您使用过Matlab的＆或| 运算符，您应该使用NumPy ufuncs logical_and / logical_or。Matlab和NumPy的＆和|之间的显着差异 运营商是：

非逻辑{0,1}输入：NumPy的输出是输入的按位AND。Matlab将任何非零值视为1并返回逻辑AND。例如，NumPy中的（3和4）是0，而在Matlab中，3和4都被认为是逻辑真，而（3和4）返回1。
优先级：NumPy的＆运算符优先于<和>之类的逻辑运算符; Matlab是相反的。
如果你知道你有布尔参数，你可以使用NumPy的按位运算符，但要注意括号，如：z =（x> 1）＆（x <2）。缺少NumPy运算符形式的logical_and和logical_or是Python设计的一个不幸结果。

重塑与线性索引：Matlab总是允许使用标量或线性索引访问多维数组，而NumPy则不然。线性索引在Matlab程序中很常见，例如矩阵上的find()返回它们，而NumPy的查找行为则不同。在转换Matlab代码时，可能需要首先将矩阵重新整形为线性序列，执行一些索引操作然后重新整形。由于重塑（通常）会在同一存储上生成视图，因此应该可以相当有效地执行此操作。请注意，在NumPy中重新整形使用的扫描顺序默认为'C'顺序，而Matlab使用Fortran顺序。如果您只是简单地转换为线性序列，那么这无关紧要。但是如果要从依赖于扫描顺序的Matlab代码转换重构，那么这个Matlab代码：z = reshape(x，3,4) 应该变成 z = x.reshape(3,4,order=’F’).copy() 。

#自定义您的环境
在MATLAB®中，可用于自定义环境的主要工具是使用您喜欢的功能的位置修改搜索路径。您可以将此类自定义项放入MATLAB将在启动时运行的启动脚本中。

NumPy，或者更确切地说是Python，具有类似的功能。

要修改Python搜索路径以包含您自己的模块的位置，请定义PYTHONPATH环境变量。
要在启动交互式Python解释器时执行特定的脚本文件，请定义PYTHONSTARTUP环境变量以包含启动脚本的名称。
与MATLAB®不同，可以立即调用路径上的任何内容，使用Python，您需要先执行“import”语句，以使特定文件中的函数可访问。

例如，您可能会创建一个如下所示的启动脚本（注意：这只是一个示例，而不是“最佳实践”的声明）：

# Make all numpy available via shorter 'np' prefix
import numpy as np
# Make all matlib functions accessible at the top level via M.func()
import numpy.matlib as M
# Make some matlib functions accessible directly at the top level via, e.g. rand(3,3)
from numpy.matlib import rand,zeros,ones,empty,eye
# Define a Hermitian function
def hermitian(A, **kwargs):
    return np.transpose(A,**kwargs).conj()
# Make some shortcuts for transpose,hermitian:
#    np.transpose(A) --> T(A)
#    hermitian(A) --> H(A)
T = np.transpose
H = hermitian

## Python最全numpy的线性代数函数功能及用法
https://www.pythonf.cn/read/97535
Python系列
numpy的线性代数函数
（1）diag
（2）dot
（3）trace
（4）det
（5）eig
（6）inv
（7）qr
（8）svd
（9）solve
（10）lstsq
感谢浏览，别忘了点个赞鼓励一下哦(^^)~~
Python系列
作为初学者，自己整理的numpy线性代数函数用法与功能，与大家分享！有帮助到您的，希望点个赞再走呗！整理不易，一起学习，共同进步啊~~
若有错误的地方欢迎留言指正！！
numpy的线性代数函数
import numpy as np    # 先导入numpy模块
（1）diag
功能：形成一个以一维数组为对角线元素的矩阵或输出矩阵的对角线元素
用法：np.diag(a)
例：

a = [1,2,3]                      # 一维数组
np.diag(a)

Out：array([[1, 0, 0],
           [0, 2, 0],
           [0, 0, 3]])           # 输出以a为对角线元素的矩阵(方阵)
b = ([1,2,3],[4,5,6],[7,8,9])    # 多维数组
np.diag(b)

Out：array([1, 5, 9])            # 输出矩阵b的对角线元素
（2）dot
功能：计算矩阵的乘法（两个数组的内积/点积）
用法：np.dot(a,b)
例：

a = ([1,2],[3,4])
b = ([5,6],[7,8])
np.dot(a,b)

Out：array([[19, 22],
            [43, 50]])
（3）trace
功能：计算矩阵的迹（对角线元素的和）
用法：np.trace(a)
例：

a = ([1,2,3],[4,5,6],[7,8,9])
np.trace(a)

Out：15
（4）det
功能：计算矩阵（方阵）的行列式值
用法：np.linalg.det(a)
例：

a = ([1,2],[3,4])
np.linalg.det(a)

Out：-2
（5）eig
功能：计算方阵的特征值和特征向量
用法：np.linalg.eig(a)
例：

a = ([1,2],[3,4])
e,v = np.linalg.eig(a)

In：e
Out： array([-0.37228132,  5.37228132])      # 特征值
In：v
Out： array([[-0.82456484, -0.41597356],
       	     [ 0.56576746, -0.90937671]])    # 特征向量
（6）inv
功能：计算矩阵的逆矩阵
用法：np.linalg.inv(a)
例：

a = ([1,2],[3,4])
np.linalg.inv(a)

Out：array([[-2. ,  1. ],
            [ 1.5, -0.5]])
（7）qr
功能：QR分解（正交分解）
用法：np.linalg.qr(a)
例：

a = ([1,2],[3,4])
q,r = np.linalg.qr(a)

In：q
Out：array([[-0.31622777, -0.9486833 ],
            [-0.9486833 ,  0.31622777]])
In：r
Out：array([[-3.16227766, -4.42718872],
         	[ 0.        , -0.63245553]])
（8）svd
功能：SVD分解（奇异值分解）
用法：np.linalg.svd(a,full_matrices=1,compute_uv=1)
例：

a = ([1,2],[3,4])
u,s,v = np.linalg.svd(a)

In：u
Out：array([[-0.40455358, -0.9145143 ],
           [-0.9145143 ,  0.40455358]])
In：s
Out：array([5.4649857 , 0.36596619])
In：v
Out：array([[-0.57604844, -0.81741556],
           [ 0.81741556, -0.57604844]])
（9）solve
功能：解形如AX=B的线性矩阵方程（计算A^(-1)B的解）
用法：np.linalg.solve(a,b)
例：

a = [[1,1,1],[ 0,2,5],[ 2,5,-1]]
b = [[6],[-4],[27]]
np.linalg.solve(a,b)	  

Out：array([[ 5.],
      		[ 3.],
      		[-2.]]) 
（10）lstsq
功能：估计线性模型中的系数（最小二乘解，求b=a*x中的a）
用法：np.linalg.lstsq(x,b)
例：

x = ([1,2],[3,4],[5,6],[7,8])
b = [-1,0.2,0.9,2.1]
np.linalg.lstsq(x,b)

Out：(array([ 1.95, -1.45]),
	  array([0.05]), 
	  2, 
	  array([14.2690955 ,  0.62682823]))
