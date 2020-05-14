from matplotlib import pyplot as plt


def plot_episode(Q, V, A, R, actions, plot_path):
    # Q, V, A and R are lists of arrays and need to be made into one big array
    fig = plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.title('Q')
    for a in range(Q.shape[1]):
        plt.plot(Q[:, a], label='Action ' + str(a))
    plt.legend()
    plt.subplot(2, 3, 2)
    plt.title('Value')
    plt.plot(V)
    plt.subplot(2, 3, 3)
    plt.title('Advance')
    plt.plot(A)
    plt.subplot(2, 3, 4)
    plt.title('action')
    plt.plot(actions, 'x')
    plt.subplot(2, 3, 5)
    plt.title('reward')
    plt.plot(R, 'x')
    fig.savefig(plot_path, format='png')
    plt.close()
