import numpy as np
import matplotlib.pyplot as plt
import util
import math

x_range = np.arange(-1, 1, 0.1)
y_range = np.arange(-1, 1, 0.1)
X_range, Y_range = np.meshgrid(x_range, y_range)

def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the prior distribution
    
    Outputs: None
    -----
    """
    mean_vec = np.array([0, 0])
    covariance_mat = np.array([[beta, 0], [0, beta]])

    gaussian_density_1d = []
    for xx_range, yy_range in zip(X_range, Y_range):
        gaussian_density_1d.append(np.array(util.density_Gaussian(mean_vec, covariance_mat, np.array(list(zip(xx_range, yy_range))))))

    plt.figure(0)
    plt.ylim(-1, 1)
    plt.xlim(-1, 1)
    plt.xlabel('a_0')
    plt.ylabel('a_1')
    plt.title('priorDistribution')

    plt.scatter([-0.1], [-0.5], color = 'red', label="true value of a")
    plt.contour(X_range, Y_range, np.array(gaussian_density_1d))
    plt.legend()

    return 

def posteriorDistribution(x,z,beta,sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from traninng set
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    
    Outputs: 
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """
    X = np.append(np.ones((x.size, 1), dtype=int), x, axis=1)
    XTX = np.matmul(X.T, X)

    sigma_over_tau_square = sigma2/(beta ** 2)
    I = np.identity(2)
   
    internal_calculation = np.linalg.inv((XTX + sigma_over_tau_square * I))
    mu = np.matmul(np.matmul(internal_calculation, X.T), z)
    mu = mu.reshape(2,)
    Cov = internal_calculation * sigma2

    gaussian_density_1d = []
    for xx_range, yy_range in zip(X_range, Y_range):
        gaussian_density_1d.append(np.array(util.density_Gaussian(mu, Cov, np.array(list(zip(xx_range, yy_range))))))

    plt.figure(x.size)
    plt.ylim(-1, 1)
    plt.xlim(-1, 1)
    plt.xlabel('a_0')
    plt.ylabel('a_1')
    plt.title('posteriorDistribution with N=' + str(x.size))

    plt.scatter([-0.1], [-0.5], color = 'red', label="true value of a")
    plt.contour(X_range, Y_range, np.array(gaussian_density_1d))
    plt.legend()
    
    return (mu,Cov)

def predictionDistribution(x_test,beta,sigma2,mu,Cov,x_train,z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results 
    
    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the prior distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot
    
    Outputs: None
    -----
    """
    predicted_y_list = []
    predicted_y_sigma = []
    for new_x in x_test:
        new_X = np.array([1, new_x])
        predicted_y_list.append(np.matmul(mu, new_X))
        predicted_y_sigma.append(math.sqrt(np.matmul(np.matmul(new_X.T, Cov), new_X) + sigma2))

    plt.figure(len(x_train) + 1) 
    plt.ylim(-4, 4)
    plt.xlim(-4, 4)
    plt.xlabel('input')
    plt.ylabel('target')
    plt.title('predictionDistribution with N=' + str(len(x_train)))
    plt.errorbar(x_test, predicted_y_list, yerr=predicted_y_sigma, color = 'k', ecolor = 'blue', fmt='o', capthick=2, label = "new inputs and predicted targets")
    plt.scatter(x, z, color = 'red', label = "training sample")
    plt.legend()
    
    return 

if __name__ == '__main__':

    # Training data
    x_train, z_train = util.get_data_in_file('training.txt')

    # New inputs for prediction 
    x_test = [x for x in np.arange(-4,4.01,0.2)]

    # Known parameters 
    sigma2 = 0.1
    beta = 1

    # Prior distribution p(a)
    priorDistribution(beta)

    # Posterior distribution p(a|x,z)
    for ns in [1, 5, 100]:
        x = x_train[0:ns]
        z = z_train[0:ns]
        (mu, Cov) = posteriorDistribution(x,z,beta,sigma2)

        # Distribution of the prediction
        predictionDistribution(x_test,beta,sigma2,mu,Cov,x,z)
    


    
