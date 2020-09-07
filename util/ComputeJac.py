import torch

def compute_jac(in_tensor, out_tensor, in_dim, out_dim):
    jac = torch.zeros(out_dim, in_dim).to(in_tensor.device)

    for i in range(out_dim):
        g = torch.zeros(out_dim).to(in_tensor.device)
        g[i] = 1.0
        grad = torch.autograd.grad(out_tensor, in_tensor, grad_outputs=g, create_graph=True)[0]
        jac[i, :] = grad
    return jac


if __name__ == "__main__":
    import tensorflow as tf
    import numpy as np

    nn = torch.nn

    def func(A, b):
        with tf.GradientTape() as tape:
            tape.watch(A)
            tape.watch(b)
            x = tf.tanh(tf.linalg.matvec(A, b))

        return x, tape.jacobian(x, b)

    rng = np.random.RandomState(0)
    dim_in = 10
    dim_out = 5

    A_np = rng.normal(size=[dim_out, dim_in])
    b_np = rng.normal(size=[dim_in])


    tf_x, tf_jac = func(tf.constant(A_np), tf.constant(b_np))

    A = torch.from_numpy(A_np)
    A.requires_grad = True
    b = torch.from_numpy(b_np)
    b.requires_grad = True
    y = torch.tanh(A @ b)

    torch_jac = compute_jac(b, y, in_dim=dim_in, out_dim=dim_out)
    torch_jac = torch_jac.detach().numpy()

    print(np.max(np.abs(torch_jac - np.array(tf_jac))))
    print("Jacobian")
    """
    print(jac)
    print("Jacobian Analyitcal")
    print(torch.diag(1 - torch.square(x)[:, 0]) @ A)
    print((1 - torch.square(x)) * A)
    #print(jac)
    """