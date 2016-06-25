cd ./gdlfiles/
mex -O gacLlinks_c.cpp
mex -O gacOnelink_c.cpp
mex -O gacPartial_sort.cpp
mex -O gacPartialMin_knn_c.cpp
mex -O gacPartialMin_triu_c.cpp
mex -O gdlInitAffinityTable_c.cpp gdlComputeAffinity.cpp
mex -O gdlInitAffinityTable_knn_c.cpp gdlComputeAffinity.cpp
mex -O gdlAffinity_c.cpp gdlComputeAffinity.cpp
mex -O gdlDirectedAffinity_c.cpp gdlComputeDirectedAffinity.cpp
mex -O gdlDirectedAffinity_batch_c.cpp gdlComputeDirectedAffinity.cpp
cd ../