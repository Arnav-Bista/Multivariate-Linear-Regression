using Plots
using CSV
using Tables
plotlyjs()
#Load all data and return them as variables
function load(path)
    df = CSV.File(path) |> Tables.matrix
    m,p = size(df)
    X = fill(1.0,m,p-1)
    X[:,2:4] = df[:,2:4]
    Y = df[:,5]
    return X,Y
end

function dnorm(X)
    mx = maximum(X)
    mn = minimum(X)
    for i in 1:size(X)[1]
        X[i] = (X[i] - mn)/(mx - mn)
    end
    return X
end

function graph_var(X,Y)
    #Graphing Variables
    tv = scatter(X[:,2],Y,xlabel = "TV Budget")
    radio = scatter(X[:,3],Y,xlabel = "Radio Budget")
    news = scatter(X[:,4],xlabel = "News Budget")
    plot(tv,radio,news, legend = false)
    ylabel!("Sales Revenue")
end

function MSE(X,Y,theta)
    m = size(Y)[1]
    return 1/m * sum((Y .- X*theta).^2)
end

function BGD(X,Y,theta)
    m = size(Y)[1]
    a = 0.00001
    iterations = 1500
    J = fill(0.0,iterations)
    for i in 1:iterations
        theta = theta - a *  2/m * (transpose(X) * (X*theta .-Y))
        J[i] = MSE(X,Y,theta)
    end
    return theta,J
end

function SGD(X,Y,theta)
    m = size(Y)[1]
    a = 0.001
    epoch = 11
    J = fill(0.0,epoch*m)
    for j in 1:epoch
        for i in 1:m
            theta = theta - a * 2/m * X[i,:] * (sum(X[i,:] .* theta) - Y[i])
            J[i + m*(j-1)] = MSE(X,Y,theta)
        end
    end
    return theta,J
end

function MBGD(X,Y,theta)
    batch_size = 23
    a = 0.0001
    m = size(Y)[1]
    num_batch = m/batch_size
    epoch = 30
    J = []
    for j in 1:epoch
        for i in 1:num_batch
            index_1 = trunc(Int,(i-1) * batch_size + 1)
            index_2 = trunc(Int,(i) * batch_size)
            theta = theta - a * 2/m * transpose(X[index_1:index_2,:]) * (X[index_1:index_2,:] * theta .- Y[index_1:index_2] )
            push!(J, MSE(X,Y,theta))
        end
    end
    return theta,J
end

function Test(X,Y,theta)
    c = 0
    for i in 1:size(Y)[1]
        t = round(sum(X[i,:] .* theta),digits = 2)
        println(Y[i], "  ", t)
        if (Y[i] - 2) < t && t < (Y[i]+2)
            c = c + 1
        end
    end
    println("Accuracy: ",round(c/size(Y)[1],digits = 2))
end
function NormalEQ(X,Y)
    return inv(transpose(X)*X) * (transpose(X)*Y)
end

function R_Squared(X,Y,theta)
    m = size(Y)[1]
    TSS = sum((Y .- sum(Y)/m).^2)
    RSS = sum((Y .- X*theta).^2)
    return (1 - (RSS/TSS))
end

X,Y = load("Advertising.csv")
theta = fill(0.0,size(X)[2])
#theta = rand(size(X)[2])
t = copy(theta)
#
#
Z = fill(0.0, 200)
for i in 2:200
    z,J = BGD(X[1:i,:],Y[1:i],t)
    # Z[i] = MSE(X,Y,z)
    Z[i] = R_Squared(X,Y,z)
end


f = open("data.txt", "w")
write(f,join(Z,","))
close(f)
plot(Z)

theta,J= MBGD(X,Y,theta)
plot(J[2:size(J)[1]],xlabel = "Iterations",ylabel = "Cost",label = "MBGD")
theta,J= BGD(X,Y,t)
plot!(J[2:size(J)[1]],xlabel = "Iterations",ylabel = "Cost",label = "BGD")
theta,J= SGD(X,Y,t)
plot!(J[2:size(J)[1]],xlabel = "Iterations",ylabel = "Cost",label = "SGD")







# t = fill(0.0,20,4)
# for i in 1:20
#     z = rand(size(X)[2])
#     t[i,:] =  BGD(X,Y,z)
# end
# std = [0.0 for i in 1:size(t)[2]]
# avg_theta = fill(0.0,4)
# for i in 1:size(t)[2]
#     s = sum(t[:,i])/20
#     std[i] = (sum((t[:,i] .- s).^2)/20).^(0.5)
#     avg_theta[i] = sum(t[:,i])/20
#     #println(avg_theta[i])
#
# end
# #println("New Cost: ",round(MSE(X,Y,theta),digits = 3))
# #print(theta)
# #println(t[:,1])
# f(x) =( 1 / (std[2] * (2 * pi)^0.5)) * exp( -(1/2) * (((x - avg_theta[2])/std[2])^2) )
#
# plot(f,0,0.2)
# #x,y = load("test.csv")
# #println("Test Cost: ",round(MSE(x,y,theta),digits = 3))
# #Test(x,y,theta)
# #graph_var(X,Y)
#
# #plot(J,xlabel = "Iterations",ylabel = "Cost", legend = false)
