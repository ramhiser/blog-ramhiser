---
title: "Setting Up a Kubernetes Cluster on AWS in 5 Minutes"
date: 2018-05-20T14:57:03-05:00
categories:
- Kubernetes
- Docker
- AWS
comments: true
---

[Kubernetes](https://kubernetes.io/) is like magic. It is a system for working with containerized applications: deployment, scaling, management, service discovery, magic. Think Docker at scale with little hassle. Despite the power of Kubernetes though, I find the [official guide](https://github.com/kubernetes/kops/blob/master/docs/aws.md) for setting up Kubernetes on AWS a bit overwhelming, so I wrote a simpler version to get started.

As a side note, AWS introduced a new serviced called [Amazon Elastic Container Service for Kubernetes](https://aws.amazon.com/eks/) -- EKS for short. But it's still in Preview mode.

Before we begin, here's a YouTube video demonstrating how to set up a Kubernetes Cluster on AWS following the instructions below:

{{< youtube RZwb6hhZvqM_44 >}}

Anyway, let's get started.

**UPDATED (28 May 2018)**: I updated the guide below to include deploying the [Kubernetes Dashboard](https://github.com/kubernetes/dashboard). I've also included a YouTube video illustrating the Dashboard installation.

***

# Prerequisites

Before setting up the Kubernetes cluster, you'll need an [AWS account](https://aws.amazon.com/account/) and an installation of the [AWS Command Line Interface](https://aws.amazon.com/cli/).

Make sure to configure the AWS CLI to use your access key ID and secret access key:

```
$ aws configure
AWS Access Key ID [None]: ramhiser_key
AWS Secret Access Key [None]: ramhiser_secret
Default region name [None]: us-east-1
Default output format [None]:
```

***

# Installing kops + kubectl

Now, to get started, let's install two Kubernetes CLI utilities:

1. [Kubernetes Operations, `kops`](https://github.com/kubernetes/kops)
2. [Kubernetes command-line tool, `kubectl`](https://kubernetes.io/docs/reference/kubectl/overview/)

On Mac OS X, we'll use [`brew`](https://brew.sh/) to install. If you're on Linux, see the [official Kops installation guide](https://github.com/kubernetes/kops#installing).

```bash
brew update && brew install kops kubectl
```

***

# Setting Up the Kubernetes Cluster

Easy enough. Now, let's set up the Kubernetes cluster.

The first thing we need to do is create an S3 bucket for `kops` to use to store the state of the Kubernetes cluster and its configuration. We'll use the bucket name `ramhiser-kops-state-store`.

```bash
$ aws s3api create-bucket --bucket ramhiser-kops-state-store --region us-east-1
```

After creating the `ramhiser-kops-state-store` bucket, let's enable versioning to revert or recover a previous state store.

```bash
$ aws s3api put-bucket-versioning --bucket ramhiser-kops-state-store  --versioning-configuration Status=Enabled
```

Before creating the cluster, let's set two environment variables: `KOPS_CLUSTER_NAME` and `KOPS_STATE_STORE`. For safe keeping you should add the following to your `~/.bash_profile` or `~/.bashrc` configs (or whatever the equivalent is if you don't use bash).

```bash
export KOPS_CLUSTER_NAME=ramhiser.k8s.local
export KOPS_STATE_STORE=s3://ramhiser-kops-state-store
```

You don't HAVE TO set the environment variables, but they are useful and referenced by `kops` commands. For example, see `kops create cluster --help`. If the the Kubernetes cluster name ends with `k8s.local`, Kubernetes will create a [gossip-based cluster](https://github.com/kubernetes/kops/blob/master/docs/aws.md#prepare-local-environment).

Now, to generate the cluster configuration:

```bash
$ kops create cluster --node-count=2 --node-size=t2.medium --zones=us-east-1a
```

**Note**: this line doesn't launch the AWS EC2 instances. It simply creates the configuration and writes to the `s3://ramhiser-kops-state-store` bucket we created above. In our example, we're creating 2 `t2.medium` EC2 work nodes in addition to a `c4.large` master instance (default).

```bash
$ kops edit cluster
```

Alternatively, you can name the cluster by appending `--name` to the command:

```bash
$ kops create cluster --node-count=2 --node-size=t2.medium --zones=us-east-1a --name chubby-bunnies
```

Now that we've generated a cluster configuration, we can edit its description before launching the instances. The config is loaded from `s3://ramhiser-kops-state-store`. You can change the editor used to edit the config by setting `$EDITOR` or `$KUBE_EDITOR`. For instance, in my `~/.bashrc`, I have `export KUBE_EDITOR=emacs`.

Time to build the cluster. This takes a few minutes to boot the EC2 instances and download the Kubernetes components.

```bash
kops update cluster --name ${KOPS_CLUSTER_NAME} --yes
```

After waiting a bit, let's validate the cluster to ensure the master + 2 nodes have launched.


```bash
$ kops validate cluster
Validating cluster ramhiser.k8s.local

INSTANCE GROUPS
NAME			ROLE	MACHINETYPE	MIN	MAX	SUBNETS
master-us-east-1a	Master	c4.large	1	1	us-east-1a
nodes			Node	t2.medium	2	2	us-east-1a

NODE STATUS
NAME				ROLE	READY
ip-172-20-34-111.ec2.internal	node	True
ip-172-20-40-24.ec2.internal	master	True
ip-172-20-62-139.ec2.internal	node	True
```

**Note**: If you ignore the message `Cluster is starting. It should be ready in a few minutes.` and validate too early, you'll get an error. Wait a little longer for the nodes to launch, and the validate step will return without error.

```bash
$ kops validate cluster
Validating cluster ramhiser.k8s.local

unexpected error during validation: error listing nodes: Get https://api-ramhiser-k8s-local-71cb48-202595039.us-east-1.elb.amazonaws.com/api/v1/nodes: EOF
```

Finally, you can see your Kubernetes nodes with `kubectl`:

```bash
$ kubectl get nodes
NAME                            STATUS    ROLES     AGE       VERSION
ip-172-20-34-111.ec2.internal   Ready     node      2h        v1.9.3
ip-172-20-40-24.ec2.internal    Ready     master    2h        v1.9.3
ip-172-20-62-139.ec2.internal   Ready     node      2h        v1.9.3
```

***

# Kubernetes Dashboard

Excellent. We have a working Kubernetes cluster deployed on AWS. At this point, we can deploy lots of things, such as [Dask and Jupyter](https://ramhiser.com/post/2018-05-28-adding-dask-and-jupyter-to-kubernetes-cluster/). For demonstration, we'll launch the [Kubernetes Dashboard](https://github.com/kubernetes/dashboard). Think UI instead of command line for managing Kubernetes clusters and applications.

To get started, let's deploy the dashboard app.

```bash
$ kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/master/src/deploy/recommended/kubernetes-dashboard.yaml

secret "kubernetes-dashboard-certs" created
serviceaccount "kubernetes-dashboard" created
role.rbac.authorization.k8s.io "kubernetes-dashboard-minimal" created
rolebinding.rbac.authorization.k8s.io "kubernetes-dashboard-minimal" created
deployment.apps "kubernetes-dashboard" created
service "kubernetes-dashboard" created
```

You can see that various items were created, including the `kubernetes-dashboard` service and app.

If the dashboard was created, how do we view it? Easy. Let's get the AWS hostname:

```bash
$ kubectl cluster-info

Kubernetes master is running at https://api-ramhiser-k8s-local-71cb48-202595039.us-east-1.elb.amazonaws.com
KubeDNS is running at https://api-ramhiser-k8s-local-71cb48-202595039.us-east-1.elb.amazonaws.com/api/v1/namespaces/kube-system/services/kube-dns:dns/proxy

To further debug and diagnose cluster problems, use 'kubectl cluster-info dump'.
```

With this hostname, open your browser to `https://api-ramhiser-k8s-local-71cb48-202595039.us-east-1.elb.amazonaws.com/ui`. (You'll need to replace the hostname with yours).

Alternatively, you can access the Dashboard UI via a proxy:

```bash
$ kubectl proxy

Starting to serve on 127.0.0.1:8001
```

Then, open your browser to [http://localhost:8001/api/v1/namespaces/kube-system/services/https:kubernetes-dashboard:/proxy/](http://localhost:8001/api/v1/namespaces/kube-system/services/https:kubernetes-dashboard:/proxy/).

At this point, you'll be prompted for a username and password.  

![login prompt](https://user-images.githubusercontent.com/261183/40863562-f2fbebf2-65b5-11e8-8c45-8e292e304d58.png)

The username is `admin`. To get the password at the CLI, type:

```
$ kops get secrets kube --type secret -oplaintext
```

After you log in, you'll see another prompt. 

![token prompt](https://user-images.githubusercontent.com/261183/40863832-e35c076c-65b6-11e8-8140-fe3262b5c47f.png)


Select **Token**. To get the **Token**, type:

```bash
kops get secrets admin --type secret -oplaintext
```

After typing in the token, you'll see the Dashboard!

![dashboard](https://user-images.githubusercontent.com/261183/40864104-fc16c386-65b7-11e8-8f04-d0fc501932f6.png)

***

# Delete the Kubernetes Cluster

When you're ready to tear down your Kubernetes cluster or if you messed up and need to start over, you can delete the cluster with a single command:

```bash
kops delete cluster --name ${KOPS_CLUSTER_NAME} --yes
```

The `--yes` argument is required to delete the cluster. Otherwise, Kubernetes will perform a dry run without deleting the cluster.
