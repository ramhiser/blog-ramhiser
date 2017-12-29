---
title: "Bye Bye, Jekyll. Hello, Hugo."
date: 2017-12-28T20:54:29-06:00
categories:
- Hugo
- Jekyll
- Go
comments: true
---

After one too many Liquid errors and all the Ruby dependency hell, I'm moving away from [Jekyll](https://jekyllrb.com/) to [Hugo](https://gohugo.io/). One less excuse to post more.

A handful of things to note about Hugo:

* Built with Go.
* Easy to use.
* It's fast. So fast. Local site rebuilds in ~100ms.
* Immediate feedback as you're writing because fast.

Getting started with Hugo is pretty easy. The [quickstart guide](https://gohugo.io/getting-started/quick-start/) is helpful.
Migrating to Hugo from Jekyll was successful and mostly uneventful.
This post started as a smoke test and a way to organize my thoughts on the migration.

### Minimal Hugo Site

Here's how I migrated. I'm on a Mac with [Homebrew](https://brew.sh/) installed. To install Hugo:

```
brew install hugo
```

Now, let's create a barebones blog and write our first post.

```
hugo new site blog-ramhiser
cd blog-ramhiser
hugo new posts/my-first-post.md
echo 'MAH FIRST POST' >> content/posts/my-first-post.md
```

I like the look and feel of [Hugo's cactus theme](https://themes.gohugo.io/cactus/).
Hugo themes play nicely with git submodules, so that's how we'll install the theme.
After cactus is loaded, copy the example `config.toml` configuration.
**NOTE**: Remove line: `themesDir = "../.."` from the config file.

```
git init
git submodule add git@github.com:digitalcraftsman/hugo-cactus-theme.git themes/hugo-cactus-theme
cp themes/hugo-cactus-theme/exampleSite/config.toml .
```

To launch the blog for the first time:

```
hugo server  
```

At this point, you'll have a stylish minimal site.

### Import from Jekyll

[Importing from Jekyll](https://gohugo.io/commands/hugo_import/) is a breeze and doesn't require a third-party tool or plugin.
From the command line, type:

```
hugo import jekyll ~/jekyll_blog/ ~/hugo_blog/
```

The target directory `~/hugo_blog/` is assumed not to exist. Once you import from Jekyll,
you can add a theme and update the `config.toml` like we did above.

Overall, the import went smoothly. I encountered a couple of [problems](https://github.com/ramhiser/blog-ramhiser/issues/1)
with weird syntax highlighting and bad list formatting. Those are minor issues though and easily fixed.

### GitHub Pages -> Netlify

For a few years now, I've been using GitHub Pages to host my blog. It's been
reliable and out of the way as good technology should be. However, I ran into some issues
a few months ago with GitHub Pages, but I think Jekyll was more to blame. At some
point though I learned that [HTTPS is not supported for GitHub Pages using custom domains](https://help.github.com/articles/securing-your-github-pages-site-with-https/). **Le sigh**.

[A friend of mine](https://yihui.name/en/) made a case for [switching to Netlify](https://yihui.name/en/2017/06/netlify-instead-of-github-pages/).
I explored that route with success. It took me 20-30 minutes with the [help of the hugo docs](https://gohugo.io/hosting-and-deployment/hosting-on-netlify/). Here's how I did it:

* Create a new GitHub repository with the Hugo blog.
* Added a [`netlify.toml` config file](https://www.netlify.com/blog/2017/04/11/netlify-plus-hugo-0.20-and-beyond/) with Hugo version 0.31.1.
* Create a Netlify account.
* Create a new Netlify site pointed at the GitHub repo.
* Update [Namecheap](https://www.namecheap.com/) DNS entries to Netlify's.
* Request a Let's Encrypt TLS certificate on the Netlify site (took a few minutes).
* Force `http` requests to redirect to `https`.

A few more details about https and Namecheap can be found [here](https://jameshfisher.com/2017/08/08/hosting-on-netlify.html).
