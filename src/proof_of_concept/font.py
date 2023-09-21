import fontforge

# Create a new font
font = fontforge.font()

# Create a new glyph with the Unicode value of the 'A' character
glyph = font.createChar(65, "A")

# Import an SVG into the glyph
glyph.importOutlines("../sample/A.svg")

# Set glyph metrics (optionally)
glyph.width = 500
glyph.vwidth = 500

f_glyph = font.createChar(70, "F")
f_glyph.importOutlines("../sample/F.svg")
f_glyph.width = 500
f_glyph.vwidth = 500

i_glyph = font.createChar(73, "I")
i_glyph.importOutlines("../sample/I.svg")
i_glyph.width = 500
i_glyph.vwidth = 500

# Create a ligature glyph. For 'f' and 'i', it would be named 'f_i'
ligature = font.createChar(-1, "f_i")

# Import an SVG into the ligature glyph
ligature.importOutlines("../sample/FI.svg")

# Define ligature relationship
font.addLookup("fi_ligature", "gsub_ligature", (), (("liga", (("latn", ("dflt")),)),))
font.addLookupSubtable("fi_ligature", "fi_ligature_subtable")

# Specify characters that will be replaced by the ligature
font["f_i"].addPosSub("fi_ligature_subtable", ("F", "I"))

# Save the font to a new file
font.generate("../sample/MyFontWithLigature.ttf")

