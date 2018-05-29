---
title: "Adding Dask and Jupyter to a Kubernetes Cluster"
date: 2018-05-28T21:02:16-05:00
categories:
- Kubernetes
- Docker
- AWS
- Dask
- Jupyter
comments: true
---

In this post, we're going to set up [Dask](https://dask.pydata.org/) and [Jupyter](http://jupyter.org/) on a [Kubernetes](https://kubernetes.io/) cluster running on AWS. If you don't have a Kubernetes cluster running, I suggest you check out [the post I wrote on setting up a Kubernetes cluster on AWS](https://ramhiser.com/post/2018-05-20-setting-up-a-kubernetes-cluster-on-aws-in-5-minutes/).


***

# Install Helm


First, let's install [Helm](https://helm.sh/), the Kubernetes package manager. On Mac OS X, we'll use [`brew`](https://brew.sh/) to install. If you're on another platform, check out the [Helm docs](https://docs.helm.sh/).

```bash
brew update && brew install kubernetes-helm
helm init
```

Once you've initialized Helm, you should see this message: `Tiller (the Helm server-side component) has been installed into your Kubernetes Cluster.`

***

# Install Dask

Next, we're going to install a Dask chart. [Kubernetes Charts](https://github.com/kubernetes/charts) are curated application definitions for Helm. To install the Dask chart, we'll update the known Charts channels before installing the stable version of Dask.

```bash
$ helm repo update
$ helm install stable/dask
Error: no available release name found
```

Okay, something's wrong. Even though [the list of stable Charts channels](https://kubernetes-charts.storage.googleapis.com) contains Dask, `helm install stable/dask` failed. What to do? Let's try `helm list`.

```bash
$ helm list
Error: configmaps is forbidden: User "system:serviceaccount:kube-system:default" cannot list configmaps in the namespace "kube-system"
```

Hmmmm. The error gives us a bit of a clue. We need to give the `serviceaccount` API permissions. Lots of details about how to do this are available in the [Kubernetes RBAC docs](https://kubernetes.io/docs/reference/access-authn-authz/rbac/#service-account-permissions). ([`RBAC = Role-based access control`](https://en.wikipedia.org/wiki/Role-based_access_control))

Fortunately, [this StackOverflow post](https://stackoverflow.com/a/46688254/234233) gives us the magic incantation to fix everything:

```bash
kubectl create serviceaccount --namespace kube-system tiller
kubectl create clusterrolebinding tiller-cluster-rule --clusterrole=cluster-admin --serviceaccount=kube-system:tiller
kubectl patch deploy --namespace kube-system tiller-deploy -p '{"spec":{"template":{"spec":{"serviceAccount":"tiller"}}}}'      
helm init --service-account tiller --upgrade
```

Let's trying installing Dask again.

```bash
$ helm install stable/dask
NAME:   running-newt
LAST DEPLOYED: Mon May 28 21:10:20 2018
NAMESPACE: default
STATUS: DEPLOYED

RESOURCES:
==> v1/Pod(related)
NAME                                          READY  STATUS             RESTARTS  AGE
running-newt-dask-jupyter-d76658fcd-xs9n7     0/1    ContainerCreating  0         0s
running-newt-dask-scheduler-5ff49977dd-vgds5  0/1    ContainerCreating  0         0s
running-newt-dask-worker-7d6dc54bff-g55v7     0/1    ContainerCreating  0         0s
running-newt-dask-worker-7d6dc54bff-jsrwt     0/1    ContainerCreating  0         0s
running-newt-dask-worker-7d6dc54bff-t2fhl     0/1    ContainerCreating  0         0s

==> v1/ConfigMap
NAME                              DATA  AGE
running-newt-dask-jupyter-config  1     0s

==> v1/Service
NAME                         TYPE          CLUSTER-IP      EXTERNAL-IP  PORT(S)                      AGE
running-newt-dask-jupyter    LoadBalancer  100.64.89.30    <pending>    80:30943/TCP                 0s
running-newt-dask-scheduler  LoadBalancer  100.71.112.161  <pending>    8786:31012/TCP,80:31121/TCP  0s

==> v1beta2/Deployment
NAME                         DESIRED  CURRENT  UP-TO-DATE  AVAILABLE  AGE
running-newt-dask-jupyter    1        1        1           0          0s
running-newt-dask-scheduler  1        1        1           0          0s
running-newt-dask-worker     3        3        3           0          0s

...

NOTE: The default password to login to the notebook server is `dask`.
```

Success! Dask is now installed on the Kubernetes cluster. Notice that Helm has given our deployment the name `running-newt`. The resources (pods and services) are all prepended with `running-newt` as well. As you can see, we launched a `dask-scheduler`, a `dask-jupyter`, and 3 `dask-worker` processes (default config). Below, we'll walk through customizing the process.

Also, notice the default Jupyter password: `dask`. We'll use it to login to our Jupyter server later.

***

# Determine AWS DNS Entry

Before we're able to work with our deployed Jupyter server, we need to determine the URL. To do this, let's start by listing all services in the namespace:

```
$ kubectl get services
NAME                          TYPE           CLUSTER-IP       EXTERNAL-IP        PORT(S)                       AGE
kubernetes                    ClusterIP      100.64.0.1       <none>             443/TCP                       18m
running-newt-dask-jupyter     LoadBalancer   100.64.89.30     a7000d65762e5...   80:30943/TCP                  1m
running-newt-dask-scheduler   LoadBalancer   100.71.112.161   a70050f1b62e5...   8786:31012/TCP,80:31121/TCP   1m
```

Notice that the `EXTERNAL-IP` displays hex values. These refer to [AWS ELB (Elastic Load Balancer)](https://aws.amazon.com/elasticloadbalancing/) entries you can find in your AWS console: `EC2 -> Load Balancers`. You can get the exposed DNS entry by matching the `EXTERNAL-IP` to the appropriate load balancer. For instance, the screenshot below shows that the DNS entry for the Jupyter node is `http://a7000d65762e511e8a4cc02a376cf962-376113908.us-east-1.elb.amazonaws.com/`.

![AWS ELB Screenshot](https://user-images.githubusercontent.com/261183/40634758-4d6402ac-62bc-11e8-9131-ea76c78dc5b0.png)

***

# Jupyter Server

Now that we have the DNS entry, let's go to the Jupyter server in the browser at: `http://a7000d65762e511e8a4cc02a376cf962-376113908.us-east-1.elb.amazonaws.com/`. The first thing you'll see is a Jupyter password prompt. Recall the default password is: `dask`.

![image](https://user-images.githubusercontent.com/261183/40634824-a6773d50-62bc-11e8-8cc6-32056d0515e0.png)

After entering the password, you'll see a running instance of [JupyterLab](https://github.com/jupyterlab/jupyterlab).

![image](https://user-images.githubusercontent.com/261183/40634892-01cd3506-62bd-11e8-98b8-ad25a7c0f75c.png)

Double-click the `examples` folder. Inside, you will see 6 notebooks, demonstrating how to use Dask.

![image](https://user-images.githubusercontent.com/261183/40634989-76fe71c8-62bd-11e8-8f71-b4ab45a45bbf.png)

The notebooks include lots of useful information, such as:

* Parallelizing Python code with Dask
* Using [Dask futures](http://dask.pydata.org/en/latest/futures.html)
* Parallelizing Pandas operations with [Dask dataframes](http://dask.pydata.org/en/latest/dataframe.html)

Awesome! Now that we have Dask on Kubernetes, we can analyze large data sets across a large cluster. Enjoy!

***

# Customize Dask Configuration

The default Helm configuration deploys 3 worker nodes, each with two cores. Each node includes a [standard Conda environment](https://conda.io/). To customize the deployment, we'll apply `helm upgrade` using a config YAML file.

Here is our `config.yaml` based on the [Dask docs](http://dask.pydata.org/en/latest/setup/kubernetes-helm.html#configure-environment):

```
worker:
  replicas: 6
  resources:
    limits:
      cpu: 2
      memory: 7.5G
    requests:
      cpu: 2
      memory: 7.5G
  env:
    - name: EXTRA_CONDA_PACKAGES
      value: numba xarray -c conda-forge
    - name: EXTRA_PIP_PACKAGES
      value: s3fs dask-ml --upgrade
jupyter:
  enabled: true
  env:
    - name: EXTRA_CONDA_PACKAGES
      value: numba xarray matplotlib -c conda-forge
    - name: EXTRA_PIP_PACKAGES
      value: s3fs dask-ml --upgrade
```

This config will increase the number of workers to 6. We'll also include some extra conda and pip packages. They need to be the same in both the Jupyter and worker environments.

If you `helm install` using this config file, you'll launch a separate cluster. So, let's use `helm upgrade` instead. To upgrade our Kubernetes cluster to use this config, type:

```bash
helm upgrade running-newt stable/dask -f config.yaml
```

After a minute or two, your cluster will be updated.

***

# Disable Jupyter Server

If you decide you'd rather run Dask only without Jupyter, that's easy to do. Simply update the config YAML with:

```
jupyter:
  enabled: false
```