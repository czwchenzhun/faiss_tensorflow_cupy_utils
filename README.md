# faiss_tensorflow_cupy_utils
Used to input tensorflow tensor (CPU and GPU) into faiss for nearest neighbor search. 
<br>
With the intervention of cupy and dlpack, the search results of faiss-gpu can be converted into tensorflow tensor without extra overhead.
Means that tensorflow tensor can be constructed directly from gpu memory pointer.
Copy the script to the project to use, you need to install cupy.

-----
用于将tensorflow张量(CPU和GPU)输入faiss进行最近邻搜索。
< br >
在cupy和dlpack的介入下，faiss-gpu的搜索结果可以转换为tensorflow张量，没有额外的开销。
意味着tensorflow张量可以直接从gpu内存指针中构造。
将脚本复制到项目中使用，需要安装cupy。