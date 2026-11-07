import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn
import os

def insert_image_after(doc, after_p, img_path, width_inches):
    new_p = doc.add_paragraph()
    new_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    new_p.paragraph_format.space_before = Pt(4)
    new_p.paragraph_format.space_after = Pt(4)
    run = new_p.add_run()
    run.font.highlight_color = WD_COLOR_INDEX.YELLOW
    run.add_picture(img_path, width=Inches(width_inches))
    after_p._p.addnext(new_p._p)
    return new_p

def embed_formula_images(doc_path):
    print(f"Embedding formula images into {doc_path}...")
    doc = docx.Document(doc_path)

    # -------------------------------------------------------------------------
    # 1. Update Numbered Equation Tables (1 to 6)
    # -------------------------------------------------------------------------
    # Table 2: Equation 1 (Grad-CAM Weight)
    t2 = doc.tables[2]
    c1 = t2.rows[0].cells[1]
    c1.text = ""
    p1 = c1.paragraphs[0]
    p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r1 = p1.add_run()
    r1.font.highlight_color = WD_COLOR_INDEX.YELLOW
    r1.add_picture('docs/formula_images/eq1_gradcam_weight.png', width=Inches(2.5))
    
    # Table 3: Equation 2 (Grad-CAM Localization Map)
    t3 = doc.tables[3]
    c2 = t3.rows[0].cells[1]
    c2.text = ""
    p2 = c2.paragraphs[0]
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = p2.add_run()
    r2.font.highlight_color = WD_COLOR_INDEX.YELLOW
    r2.add_picture('docs/formula_images/eq2_gradcam_map.png', width=Inches(2.6))

    # Metric Equation Tables (Indices 6, 7, 8, 9)
    # Table 6: Equation 3 (Accuracy)
    t6 = doc.tables[6]
    c3 = t6.rows[0].cells[1]
    c3.text = ""
    p3 = c3.paragraphs[0]
    p3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r3 = p3.add_run()
    r3.font.highlight_color = WD_COLOR_INDEX.YELLOW
    r3.add_picture('docs/formula_images/eq3_accuracy.png', width=Inches(2.8))

    # Table 7: Equation 4 (Precision)
    t7 = doc.tables[7]
    c4 = t7.rows[0].cells[1]
    c4.text = ""
    p4 = c4.paragraphs[0]
    p4.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r4 = p4.add_run()
    r4.font.highlight_color = WD_COLOR_INDEX.YELLOW
    r4.add_picture('docs/formula_images/eq4_precision.png', width=Inches(2.0))

    # Table 8: Equation 5 (Recall)
    t8 = doc.tables[8]
    c5 = t8.rows[0].cells[1]
    c5.text = ""
    p5 = c5.paragraphs[0]
    p5.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r5 = p5.add_run()
    r5.font.highlight_color = WD_COLOR_INDEX.YELLOW
    r5.add_picture('docs/formula_images/eq5_recall.png', width=Inches(1.8))

    # Table 9: Equation 6 (F1-score)
    t9 = doc.tables[9]
    c6 = t9.rows[0].cells[1]
    c6.text = ""
    p6 = c6.paragraphs[0]
    p6.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r6 = p6.add_run()
    r6.font.highlight_color = WD_COLOR_INDEX.YELLOW
    r6.add_picture('docs/formula_images/eq6_f1.png', width=Inches(2.8))

    print("Updated 6 numbered equation tables with high-res formula images.")

    # -------------------------------------------------------------------------
    # 2. Embed In-Text Mathematical Formula Images
    # -------------------------------------------------------------------------
    for i, p in enumerate(doc.paragraphs):
        # Section 3.2: ImageNet standardization
        if "x_hat_(c, i, j) = (x_(c, i, j) - mu_c) / sigma_c" in p.text:
            insert_image_after(doc, p, 'docs/formula_images/eq_standardization.png', width_inches=1.5)
            print("Embedded eq_standardization.png in Section 3.2.")
            break

    for i, p in enumerate(doc.paragraphs):
        # Section 3.2.1: Deterministic Sampling Index
        if "i_k = floor( k * (N - 1) / (T - 1) )" in p.text:
            insert_image_after(doc, p, 'docs/formula_images/eq_sampling.png', width_inches=3.0)
            print("Embedded eq_sampling.png in Section 3.2.1.")
            break

    for i, p in enumerate(doc.paragraphs):
        # Section 3.4.1: Attention Energy
        if "u_t = tanh( W_a * h_t + b_a )" in p.text:
            insert_image_after(doc, p, 'docs/formula_images/eq_attention_energy.png', width_inches=2.2)
            print("Embedded eq_attention_energy.png in Section 3.4.1.")
            break

    for i, p in enumerate(doc.paragraphs):
        # Section 3.4.1: Attention Weights
        if "alpha_t = exp( v_a^T * u_t )" in p.text:
            insert_image_after(doc, p, 'docs/formula_images/eq_attention_weights.png', width_inches=3.8)
            print("Embedded eq_attention_weights.png in Section 3.4.1.")
            break

    for i, p in enumerate(doc.paragraphs):
        # Section 3.5: Grad-CAM min-max normalization
        if "H^c(x, y) = ( L_Grad-CAM^c(x, y) - min(L_Grad-CAM^c) )" in p.text:
            insert_image_after(doc, p, 'docs/formula_images/eq_gradcam_norm.png', width_inches=3.0)
            print("Embedded eq_gradcam_norm.png in Section 3.5.")
            break

    for i, p in enumerate(doc.paragraphs):
        # Section 4.3: Loss image
        if "L_image = - 1/B_img * sum_(i=1)^B_img" in p.text:
            insert_image_after(doc, p, 'docs/formula_images/eq_loss_image.png', width_inches=4.5)
            print("Embedded eq_loss_image.png in Section 4.3.")
            break

    for i, p in enumerate(doc.paragraphs):
        # Section 4.3: Loss video
        if "L_video = - 1/B_vid * sum_(j=1)^B_vid" in p.text:
            insert_image_after(doc, p, 'docs/formula_images/eq_loss_video.png', width_inches=4.5)
            print("Embedded eq_loss_video.png in Section 4.3.")
            break

    doc.save(doc_path)
    print(f"Successfully saved {doc_path} with all embedded formula images!")

if __name__ == "__main__":
    embed_formula_images("Article_Journal_DD.docx")
    embed_formula_images("Article_Journal_DD_revised.docx")
