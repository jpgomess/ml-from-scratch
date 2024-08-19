
import matplotlib.pyplot as plt
import ipywidgets as widgets
import numpy as np

from ipywidgets import interact

def interactive_plot(X, y, arrow_length):

    w0, w1 = 0, 0

    W0 = np.linspace(-100,100,25)
    W1 = np.linspace(-100,100,25)
    W0, W1 = np.meshgrid(W0, W1)

    error_surface = ((y - (W1.reshape(-1,1) * X.T + W0.reshape(-1,1)))**2).mean(axis=1).reshape(25, 25)
    dj_dw0_field = - ((y - (W1.reshape(-1,1) * X + W0.reshape(-1,1))).mean(axis=1) * 2).reshape(25,25)
    dj_dw1_field = - (((y - (W1.reshape(-1,1) * X + W0.reshape(-1,1))) * X).mean(axis=1) * 2).reshape(25,25)

    # Definindo os limites do eixo y
    max_error_surface = np.max(error_surface)
    ylim = (-max_error_surface/10, max_error_surface)
    xytext = (0, max_error_surface - max_error_surface/10)

    fig = plt.figure(figsize=(10,5))

    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233, projection='3d')
    ax4 = fig.add_subplot(223)
    ax5 = fig.add_subplot(224, sharey=ax4)

    fig.tight_layout()
    # plt.show()

    def update(w0, w1):

        # Cálculo do MSE no ponto (w0,w1)
        y_ = w0 + w1 * X
        error = ((y - y_)**2).mean()

        # Calculando o MSE nas direções de w0 e w1
        w0_line = W0[0]
        w1_line = W1[:,0]
        error_line0 = ((y - (w1 * X.T + w0_line.reshape(-1,1)))**2).mean(axis=1)
        error_line1 = ((y - (w1_line.reshape(-1,1) * X.T + w0))**2).mean(axis=1)

        # Calculando o gradiente no ponto (w0, w1)
        dj_dw0 = - 2 * (y - y_).mean()
        dj_dw1 = - 2 * ((y - y_) * X).mean()

        # Cálculo dos parâmetros das setas
        norm0 = np.sqrt(1**2 + dj_dw0**2)
        dw0 = - norm0 * dj_dw0 / abs(dj_dw0) * arrow_length
        dj0 = - abs(dj_dw0 * norm0) * arrow_length

        norm1 = np.sqrt(1**2 + dj_dw1**2)
        dw1 = - norm1 * dj_dw1 / abs(dj_dw1) * arrow_length
        dj1 = - abs(dj_dw1 * norm1) * arrow_length

        # Plot
        ax1.clear()
        ax1.scatter(X, y, edgecolor='black')
        ax1.plot(X, y_, color='red', linewidth=1)
        ax1.vlines(X, ymin=y_, ymax=y, linestyle='--', color='orange')
        ax1.set_xlabel(r'$x$')
        ax1.set_ylabel(r'$y$')

        ax2.clear()
        ax2.contourf(W0, W1, error_surface, 100, cmap='coolwarm', zorder=1)
        ax2.scatter(w0, w1, ec='black', s=80, color='yellow', zorder=3)
        ax2.hlines(w1, w0_line.min(), w0_line.max(), zorder=2, color='black', linestyle='--')
        ax2.vlines(w0, w1_line.min(), w1_line.max(), zorder=2, color='black', linestyle='--')
        ax2.set_xlabel(r'$\beta_0$')
        ax2.set_ylabel(r'$\beta_1$')

        ax3.clear()
        ax3.quiver(W0, W1, np.zeros_like(W0), dj_dw0_field, dj_dw1_field, np.ones_like(W0), length=arrow_length/5, zorder=0)
        ax3.plot_surface(W0, W1, error_surface, cmap='coolwarm', linewidth=0, zorder=2)
        ax3.plot(w0_line, np.ones(len(w1_line))*w1, error_line0, linewidth=1, color='black', linestyle='--', zorder=3)
        ax3.plot(np.ones(len(w0_line))*w0, w1_line, error_line1, linewidth=1, color='black', linestyle='--', zorder=3)
        ax3.plot(w0, w1, error, marker='o', color='yellow', zorder=4)
        ax3.set_xlabel(r'$\beta_0$')
        ax3.set_ylabel(r'$\beta_1$')
        ax3.set_zlabel(r'MSE')

        ax4.clear()
        ax4.plot(w0_line, error_line0, color='black', linestyle='--', zorder=1)
        ax4.annotate('', xy=(w0 + dw0, error + dj0), xytext=(w0, error), arrowprops={"width":3, "headwidth":6, 'headlength':6, 'color':'orange', 'ec':'black'}, zorder=2)
        ax4.annotate(r'$\frac{\nabla J}{\beta_0} = $' + str(round(dj_dw0, 2)), xy=(0,0), xytext=xytext, horizontalalignment='center', verticalalignment='top', fontsize=15)
        ax4.scatter(w0, error, c='yellow', s=40, ec='black', zorder=3)
        ax4.set_ylim(ylim)
        ax4.set_xlabel(r'$\beta_0$')
        ax4.set_ylabel('MSE')

        ax5.clear()
        ax5.plot(w1_line, error_line1, color='black', linestyle='--', zorder=1)
        ax5.annotate('', xy=(w1 + dw1, error + dj1), xytext=(w1, error), arrowprops={"width":3, "headwidth":6, 'headlength':6, 'color':'orange', 'ec':'black'}, zorder=2)
        ax5.annotate(r'$\frac{\nabla J}{\beta_1} = $' + str(round(dj_dw1, 2)), xy=(0,0), xytext=xytext, horizontalalignment='center', verticalalignment='top', fontsize=15)
        ax5.scatter(w1, error, c='yellow', s=40, ec='black', zorder=3)
        ax5.set_xlabel(r'$\beta_1$')

    interact(
        update,
        w0=widgets.FloatSlider(value=w0, min=W0.min(), max=W0.max(), step=1e-3),
        w1=widgets.FloatSlider(value=w1, min=W1.min(), max=W1.max(), step=1e-3, readout=True, readout_format='.3f')
    )

