import numpy as np
import matplotlib.pyplot as plt

##Interleaved Filter 1
T = 100000
K = 10
delta = 1 / (T*K**2)
mus1 = [0.1*i for i in range(1,11)]
mus3 = [0.586, 0.537, 0.497, 0.983, 0.392, 0.412, 0.005, 0.657, 0.940,0.242] #random

def logistic(b_,b,mus):
    prob = mus[b_] / (mus[b_] + mus[b])
    return prob

Mus = [mus1,mus2]
Regret = {}

i = 0
for mus in Mus:
    Regret[i] = {}
    b_star = np.argmax(mus)
    Regret[i] = {}
    Regret[i]['strong'] = np.zeros((50,T))
    Regret[i]['weak'] = np.zeros((50,T))
    for iteration in range(50):
        W = list(range(K))
        t = 0
        P = np.zeros((K,K))
        Wins = np.zeros((K,K))
        C_low = np.zeros((K,K))
        C_high = np.zeros((K,K))
        n_compares = np.zeros((K,K))
        b_hat = np.random.choice(W,1)[0]
        W.remove(b_hat)
        b_star = np.argmax(mus)
        while W != [] and t<T:
            for b in W:
                if t>=T:
                    break
                prob = logistic(b_hat,b,mus)
                eps1 = logistic(b_star,b_hat,mus) - 1/2
                eps2 = logistic(b_star,b,mus)- 1/2

                winner = np.random.choice([b_hat,b],p = [prob,1-prob])
                Regret[i]['strong'][iteration,t] = eps1+eps2
                Regret[i]['weak'][iteration,t] = min(eps1,eps2)


                t += 1
                n_compares[b_hat,b] += 1
                n_compares[b,b_hat] += 1
                if winner == b_hat:
                    Wins[b_hat,b] += 1
                else:
                    Wins[b,b_hat] += 1
                P[b_hat,b] = Wins[b_hat,b] / n_compares[b_hat,b]
                ct = np.sqrt(np.log(1/delta) / n_compares[b_hat,b])
                C_low[b_hat,b] = P[b_hat,b] - ct
                C_high[b_hat,b] = P[b_hat,b] + ct

            for b in W:
                if P[b_hat,b] > 0.5 and C_low[b_hat,b] > 0.5:
                    W.remove(b)

            for b in W:
                if P[b_hat,b] < 0.5 and C_high[b_hat,b] < 0.5:
                    b_hat = b
                    W.remove(b_hat)
                    break

    i+=1

##Plot Regret
def plot_regret_with_error_bars(regret,mus):
    t_range = list(range(1,T+1))
    eps12 = max(mus) / (max(mus) + np.sort(mus)[-2]) - 1/2
    regret_bound = [K*np.log(K)*np.log(t_) / eps12 for t_ in range(1,T+1)]
    low_error_strong = (np.std(np.cumsum(regret['strong'],axis=1),axis=0)).tolist()
    up_error_strong = (np.std(np.cumsum(regret['strong'],axis=1),axis=0)).tolist()
    plt.figure(figsize=(9,6))
    plt.errorbar(t_range,y=np.mean(np.cumsum(regret['strong'],axis=1),axis=0),yerr=[low_error_strong,up_error_strong],
                 c='0.8',alpha=0.5,label='+/- 1 std deviation')
    plt.plot(np.mean(np.cumsum(regret['strong'],axis=1),axis=0),c='C0',label='Mean Strong Regret')
    low_error_weak = (np.std(np.cumsum(regret['weak'],axis=1),axis=0)).tolist()
    up_error_weak = (np.std(np.cumsum(regret['weak'],axis=1),axis=0)).tolist()
    plt.errorbar(t_range,y=np.mean(np.cumsum(regret['weak'],axis=1),axis=0),yerr=[low_error_weak,up_error_weak],
                 c='0.7',alpha=0.5,label='+/- 1 std deviation')
    plt.plot(np.mean(np.cumsum(regret['weak'],axis=1),axis=0),c='C1',label='Mean Weak Regret')
    plt.plot(t_range,regret_bound,label='Regret Bound',c='C3')
    plt.xlabel('Time steps')
    plt.ylabel('Regret over time')
    plt.legend()
    plt.show()

i = 0
for mus in [mus1,mus2]:
    regret = Regret[i]
    plot_regret_with_error_bars(regret,mus)
    i+=1

## Interleaved Filter 2
T = 100000
K = 10
delta = 1 / (T*K**2)
mus1 = [0.1*i for i in range(1,11)]
mus3 = [0.586, 0.537, 0.497, 0.983, 0.392, 0.412, 0.005, 0.657, 0.940,0.242] #random

def logistic(b_,b,mus):
    prob = mus[b_] / (mus[b_] + mus[b])
    return prob

Mus = [mus1,mus2]
Regret = {}

i = 0
for mus in Mus:
    Regret[i] = {}
    b_star = np.argmax(mus)
    Regret[i] = {}
    Regret[i]['strong'] = np.zeros((50,T))
    Regret[i]['weak'] = np.zeros((50,T))
    for iteration in range(50):
        W = list(range(K))
        t = 0
        P = np.zeros((K,K))
        Wins = np.zeros((K,K))
        C_low = np.zeros((K,K))
        C_high = np.zeros((K,K))
        n_compares = np.zeros((K,K))
        b_hat = np.random.choice(W,1)[0]
        W.remove(b_hat)
        b_star = np.argmax(mus)
        while W != [] and t<T:
            for b in W:
                if t>=T:
                    break
                prob = logistic(b_hat,b,mus)
                eps1 = logistic(b_star,b_hat,mus) - 1/2
                eps2 = logistic(b_star,b,mus)- 1/2

                winner = np.random.choice([b_hat,b],p = [prob,1-prob])
                Regret[i]['strong'][iteration,t] = eps1+eps2
                Regret[i]['weak'][iteration,t] = min(eps1,eps2)


                t += 1
                n_compares[b_hat,b] += 1
                n_compares[b,b_hat] += 1
                if winner == b_hat:
                    Wins[b_hat,b] += 1
                else:
                    Wins[b,b_hat] += 1
                P[b_hat,b] = Wins[b_hat,b] / n_compares[b_hat,b]
                ct = np.sqrt(np.log(1/delta) / n_compares[b_hat,b])
                C_low[b_hat,b] = P[b_hat,b] - ct
                C_high[b_hat,b] = P[b_hat,b] + ct

            for b in W:
                if P[b_hat,b] > 0.5 and C_low[b_hat,b] > 0.5:
                    W.remove(b)

            for b in W:
                if P[b_hat,b] < 0.5 and C_high[b_hat,b] < 0.5:
                    for b_ in W:
                        if P[b_hat,b_] > 0.5:
                            W.remove(b_)
                    b_hat = b
                    W.remove(b_hat)
                    break

    i+=1

##Plotting regret
i = 0
for mus in [mus1,mus2]:
    regret = Regret[i]
    plot_regret_with_error_bars(regret,mus)
    i+=1


##Implementing IF2 with values given in paper2
T = 32000
K = 6
Mus = [[0.8,0.2,0.2,0.2,0.2,0.2],
       [0.8,0.7,0.2,0.2,0.2,0.2],
       [0.8,0.7,0.7,0.2,0.2,0.2],
       [0.8,0.7,0.575,0.45,0.325,0.2],
       [0.8,0.7,0.512,0.374,0.274,0.2]]

def natural(b_,b,mus):
    prob = mus[b_] / (mus[b_] + mus[b])
    return prob

def linear(b_,b,mus):
    prob = (1 + mus[b_] - mus[b]) / 2
    return prob

def logit(b_,b,mus):
    prob = 1 / (1 + np.exp(mus[b] - mus[b_]))
    return prob

delta = 1 / (T*K**2)
b_star = 0
Regret = {}
for iter_ in range(50):
    Regret[iter_] = {}
    for func in ['linear','natural','logit']:
        Regret[iter_][func] = []
        i = 0
        for mus in Mus:
            Regret[iter_][func].append([])
            W = list(range(K))
            b_hat = np.random.choice(W,1)[0]
            t = 0
            P = np.zeros((K,K))
            Wins = np.zeros((K,K))
            C_low = np.zeros((K,K))
            C_high = np.zeros((K,K))
            n_compares = np.zeros((K,K))

            while W != [] and t<32000:
                for b in W:
                    if func == 'linear':
                        prob = linear(b_hat,b,mus)
                        eps1 = linear(b_star,b_hat,mus) - 1/2
                        eps2 = linear(b_star,b,mus) - 1/2
                    elif func == 'natural':
                        prob = natural(b_hat,b,mus)
                        eps1 = natural(b_star,b_hat,mus) - 1/2
                        eps2 = natural(b_star,b,mus) - 1/2
                    else:
                        prob = logit(b_hat,b,mus)
                        eps1 = logit(b_star,b_hat,mus) - 1/2
                        eps2 = logit(b_star,b,mus) - 1/2

                    winner = np.random.choice([b_hat,b],p = [prob,1-prob])
                    t += 1

                    Regret[iter_][func][i].append((eps1+eps2)/2)
                    n_compares[b_hat,b] += 1
                    n_compares[b,b_hat] += 1
                    if winner == b_hat:
                        Wins[b_hat,b] += 1
                    else:
                        Wins[b,b_hat] += 1
                    P[b_hat,b] = Wins[b_hat,b] / n_compares[b_hat,b]
                    ct = np.sqrt(np.log(1/delta) / n_compares[b_hat,b])
                    C_low[b_hat,b] = P[b_hat,b] - ct
                    C_high[b_hat,b] = P[b_hat,b] + ct

                for b in W:
                    if P[b_hat,b] > 0.5 and C_low[b_hat,b] > 0.5:
                        W.remove(b)

                for b in W:
                    if P[b_hat,b] < 0.5 and C_high[b_hat,b] < 0.5:
                        for b_ in W:
                            if P[b_hat,b_] > 0.5:
                                W.remove(b_)
                        b_hat = b
                        W.remove(b_hat)
                        break
            i += 1
