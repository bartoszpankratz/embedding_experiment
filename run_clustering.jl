#run_clustering.jl embed_file g_name g_cl_name res_dir seed 

julia_path = "julia-1.7.2/bin/julia"
embed_file = ARGS[1]
params = split(embed_file[1:end-4], "_")
embed_name = split(params[1],"/")[end]

n = parse(Int,params[2])
ξ = parse(Float64,params[3])
γ = parse(Float64,params[4])
β = parse(Float64,params[5])
min_deg = parse(Int,params[6])
dims = parse(Int,params[7])

embed_params = ""
if embed_name == "node2vec" 
    embed_params = "p: $(params[8]), q: $(params[9])"
elseif embed_name == "SDNE"
    embed_params = "ℓ: $(params[8]), β: $(params[9])"
elseif embed_name == "GraRep"
    embed_params = "k: $(params[8])"
elseif embed_name == "HOPE"
    embed_params = params[8]
end



g_name = ARGS[2]
g_cl_name = ARGS[3]
res_dir = ARGS[4]
seed = parse(Int, ARGS[5])

using Random, Base.Iterators
using PyCall, Clustering
 
Random.seed!() = seed

nx = pyimport("networkx");
community_louvain = pyimport("community"); 
gmm = pyimport("sklearn.mixture")
hdbscan = pyimport("hdbscan")

#embedding parameters:

#K-Means and Gaussian Mixture Model:
n_clusters = n_clusters = [Int(ceil(n/k)) for k in 2:2:32]

#HDBSCAN:
min_samples = 1:10

#create results file:
mkpath(res_dir)
res_name = res_dir * "results_$(n)_$(ξ)_$(γ)_$(β)_$(min_deg).dat"

#read graph:
G = nx.read_edgelist(g_name)
nodeslist = collect(nx.nodes(G));

gt_partition = [parse.(Int,e2) - 1 for (e1,e2) in split.(readlines(g_cl_name))]
louvain_partition = collect(values(community_louvain.best_partition(G,random_state = seed)))
louvain_partition = convert(Array{Int,1}, louvain_partition)

#Read embedding:
embed = transpose(reduce(hcat,[parse.(Float64,x) for x in split.(readlines(embed_file)[2:end])]))
embed = embed[sortperm(embed[:, 1]), :][:,2:end];

#CGE score:
CGE_score = readchomp(`$julia_path CGE_CLI.jl -g $(g_name) -e $(embed_file) -c $(g_cl_name) --seed $(seed)`)
CGE_score = [parse(Float64, x)  for x in split.(CGE_score[2:end-1], ", ")]

#Mini Batch K-Means:
res = []
for k in n_clusters
    labels = kmeans(transpose(embed),k).assignments
    d = Dict(nodeslist[i] => labels[i] for i = 1:length(labels))
    partition = community_louvain.best_partition(G,d,random_state = seed)
    mod = community_louvain.modularity(partition,G)
    partition = convert(Array{Int,1}, collect(values(partition)))
    ami_gt = mutualinfo(gt_partition, partition)
    ami_louvain = mutualinfo(louvain_partition, partition)
    push!(res, (mod, k, ami_gt, ami_louvain))
end
best_res = sort(res, by = first, rev = true)[1]
open(res_name, "a") do io
    println(io, join(vcat([n,ξ,γ,β,min_deg,
                    embed_name,dims, embed_params, 
            "K-Means","k: $(best_res[2])",best_res[1], best_res[3], best_res[4]], CGE_score),";"))
end


#HDBSCAN
res = []
for ms in min_samples
    c = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=ms).fit(embed)
    remain_clusters =  collect((length(unique(c.labels_)) - 1):length(c.labels_))
    labels = [c.labels_[i] != -1  ? c.labels_[i] : popfirst!(remain_clusters)
              for i = 1:length(c.labels_)]
    d = Dict(nodeslist[i] => labels[i] for i = 1:length(labels))
    partition = community_louvain.best_partition(G,d, random_state = seed)
    mod = community_louvain.modularity(partition,G)
    partition = convert(Array{Int,1}, collect(values(partition)))
    ami_gt = mutualinfo(gt_partition, partition)
    ami_louvain = mutualinfo(louvain_partition, partition)
    push!(res, (mod, ms, ami_gt, ami_louvain))
end
best_res = sort(res, by = first, rev = true)[1]
open(res_name, "a") do io
    println(io, join(vcat([n,ξ,γ,β,min_deg,
                    embed_name,dims, embed_params, 
            "HDBSCAN","min_samples: $(best_res[2])",best_res[1], best_res[3], best_res[4]], CGE_score),";"))
end


#Gaussian Mixture Models
res = []
for k in n_clusters
    c = gmm.GaussianMixture(n_components=k, random_state = seed).fit(embed)
    labels = c.predict(embed)
    d = Dict(nodeslist[i] => labels[i] for i = 1:length(labels))
    partition = community_louvain.best_partition(G,d, random_state = seed)
    mod = community_louvain.modularity(partition,G)
    partition = convert(Array{Int,1}, collect(values(partition))) 
    ami_gt = mutualinfo(gt_partition, partition)
    ami_louvain = mutualinfo(louvain_partition, partition)
    push!(res, (mod, k, ami_gt, ami_louvain))
end
best_res = sort(res, by = first, rev = true)[1]
open(res_name, "a") do io
    println(io, join(vcat([n,ξ,γ,β,min_deg,
                    embed_name,dims, embed_params, 
            "GMM","k: $(best_res[2])",best_res[1], best_res[3], best_res[4]], CGE_score),";"))
end

