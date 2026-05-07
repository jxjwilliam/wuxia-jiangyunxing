## Chat with Claude

```text
I have a pdf, each page includes: left side text, right side image, as attached. can you analyse and find a solution:

1. extract the left-side text into a small txt file
2. extract the right-side image into a png/jpg/... file
3. make the 2 files as pairs into a folder which name to be page number or more advanced: far-left chinese title if you can
4. there should be a lot of such folders, make them under 1 parent folder: wuxia ask me questions if have then to start
```

```text
1. the pdf is large, 39mb, each page of the pdf is like the above attached: left-side text, right-side chinese traditional comic image
2. yes. but with some exception: e.g. start from page 6. that means skip from page 1 to page 6. hope this makes things simpler.
3. 第九回_鐵槍破犁
4. I have no idea, let's start simpier - extract text directly. if possible translate the traditional chinese into simple chinese.
5. you recommend. important: the image will be used to re-process, image-to-video in different tools or AI.
```

## Resources

- pdf: data/jiang-yun-xing.pdf
- implementation plan: docs/wuxia_extraction_plan.md
