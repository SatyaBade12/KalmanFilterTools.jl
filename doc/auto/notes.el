(TeX-add-style-hook
 "notes"
 (lambda ()
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "amsmath"))
 :latex)

