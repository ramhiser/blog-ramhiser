---
title: "Bye Bye, Jekyll. Hello, Hugo."
date: 2017-12-28T20:54:29-06:00
draft: true
---

After one too many Liquid errors and all the Ruby dependency hell, I'm moving away from [Jekyll](https://jekyllrb.com/) to [Hugo](https://gohugo.io/). A handful of things to note about Hugo:

* It's fast. So fast. I save a file, the site rebuilds in milliseconds.
* Built in Go.
* Easy to use.

[Getting started with Hugo](https://gohugo.io/getting-started/quick-start/) is pretty easy. Here's how I migrated.

I'm on a Mac with [Homebrew](https://brew.sh/) installed:

```
brew install hugo
```

Now, let's create a barebones blog and write our first post.

```
hugo new site ramhiser-blog
cd ramhiser-blog
hugo new posts/my-first-post.md
echo 'MAH FIRST POST' >> content/posts/my-first-post.md
```

I like the look and feel of [Hugo's cactus theme](https://themes.gohugo.io/cactus/), so that's what I'm using.

```
git init
git submodule add git@github.com:digitalcraftsman/hugo-cactus-theme.git themes/hugo-cactus-theme
# NOTE: Remove line: themesDir = "../.."
cp themes/hugo-cactus-theme/exampleSite/config.toml .
```

To launch the blog for the first time:

```
hugo server  
```

At this point, you'll have a stylish minimal site.

[Importing from Jekyll](https://gohugo.io/commands/hugo_import/) is a breeze:

```
hugo import jekyll ~/jekyll_blog/ ~/hugo_blog/
```

I use GitHub Pages for hosting the blog. It turns out it's pretty simple [to deploy Hugo on GitHub](https://gohugo.io/hosting-and-deployment/hosting-on-github/):

1. Add `publishDir = "docs"` to your `config.toml`
2. In your GitHub project, go to Settings -> GitHub Pages.
3. Choose `master branch /docs folder` from Source (requires that you have a `docs/` folder already.)

Pretty easy, huh?
