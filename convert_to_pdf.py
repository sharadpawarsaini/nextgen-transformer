import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#718096"))
        
        # Top Header line
        self.setLineWidth(0.5)
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.line(54, 11 * 72 - 36, 8.5 * 72 - 54, 11 * 72 - 36)
        self.drawString(54, 11 * 72 - 30, "nnFormer 3D Medical Image Segmentation — End-to-End Pipeline Execution Report")

        # Bottom Footer line
        self.line(54, 45, 8.5 * 72 - 54, 45)
        self.setFont("Helvetica", 8)
        self.drawString(54, 32, "Confidential & Automated Technical Execution Report")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * 72 - 54, 32, page_str)
        self.restoreState()


def create_pdf(filename):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    primary_color = colors.HexColor("#1A365D")
    secondary_color = colors.HexColor("#2B6CB0")
    dark_neutral = colors.HexColor("#2D3748")

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=primary_color,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=secondary_color,
        spaceAfter=15
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=primary_color,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=dark_neutral,
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'BulletText',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=dark_neutral
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.white
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11.5,
        textColor=dark_neutral
    )

    elements = []

    # Title & Subtitle
    elements.append(Paragraph("nnFormer Full Pipeline Execution Report", title_style))
    elements.append(Paragraph("Automated 3D Inter-slice and Intra-slice Medical Image Segmentation", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceAfter=12))

    # Executive Summary Box
    summary_text = (
        "<b>Executive Summary:</b> The repository <b>282857341/nnFormer</b> was cloned to "
        "<code>c:\\Users\\shara\\Desktop\\nnFormer</code>, configured in a dedicated Python 3.10 virtual environment "
        "with PyTorch CUDA acceleration on an NVIDIA GeForce MX330 GPU. All Windows OS compatibility barriers, "
        "multiprocessing pickling errors, and generator syntax deprecations were systematically resolved. "
        "The end-to-end pipeline executed to complete success (Exit Code 0), covering synthetic 3D ACDC MRI "
        "dataset generation, experiment planning, 3D Swin-Transformer network training, 3D sliding window inference, "
        "and multi-class Dice metric evaluation."
    )
    
    summary_p = Paragraph(summary_text, ParagraphStyle('SummaryP', parent=body_style, fontSize=9, leading=13))
    summary_table = Table([[summary_p]], colWidths=[504])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EBF8FF")),
        ('BORDER', (0, 0), (-1, -1), 1, colors.HexColor("#BEE3F8")),
        ('PADDING', (0, 0), (-1, -1), 10),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 10))

    # Environment Setup Section
    elements.append(Paragraph("1. Environment & Hardware Specifications", h1_style))
    env_data = [
        [Paragraph("<b>Component</b>", table_header), Paragraph("<b>Specification / Configuration</b>", table_header)],
        [Paragraph("Repository Base", table_cell_bold), Paragraph("<code>c:\\Users\\shara\\Desktop\\nnFormer</code>", table_cell)],
        [Paragraph("Python Environment", table_cell_bold), Paragraph("Python 3.10.20 (Virtual Environment `.venv`)", table_cell)],
        [Paragraph("PyTorch Engine", table_cell_bold), Paragraph("PyTorch 2.5.1+cu121 / TorchVision 0.20.1+cu121", table_cell)],
        [Paragraph("Compute Device", table_cell_bold), Paragraph("NVIDIA GeForce MX330 GPU (CUDA Enabled & Active)", table_cell)],
        [Paragraph("Dataset Task", table_cell_bold), Paragraph("Task001_ACDC (Automated Cardiac Diagnosis Challenge 3D MRI)", table_cell)],
    ]
    env_table = Table(env_data, colWidths=[150, 354])
    env_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(env_table)
    elements.append(Spacer(1, 10))

    # Pipeline Execution Breakdown Table
    elements.append(Paragraph("2. End-to-End Pipeline Steps Breakdown", h1_style))
    pipeline_data = [
        [Paragraph("<b>Step</b>", table_header), Paragraph("<b>Pipeline Stage</b>", table_header), Paragraph("<b>Status</b>", table_header), Paragraph("<b>Execution Details & Output Artifacts</b>", table_header)],
        [
            Paragraph("Step 1", table_cell_bold),
            Paragraph("Dataset Setup", table_cell),
            Paragraph("<font color='#2F855A'><b>SUCCESS</b></font>", table_cell),
            Paragraph("Decathlon 3D NIfTI formatting (`imagesTr`, `labelsTr`, `imagesTs`, `labelsTs`). 4 3D volumes created.", table_cell)
        ],
        [
            Paragraph("Step 2", table_cell_bold),
            Paragraph("Experiment Planning", table_cell),
            Paragraph("<font color='#2F855A'><b>SUCCESS</b></font>", table_cell),
            Paragraph("`nnFormer_plan_and_preprocess` generated plans (`nnFormerPlansv2.1_plans_3D.pkl`). Crop patch: 14x160x160.", table_cell)
        ],
        [
            Paragraph("Step 3", table_cell_bold),
            Paragraph("Network Training", table_cell),
            Paragraph("<font color='#2F855A'><b>SUCCESS</b></font>", table_cell),
            Paragraph("`nnFormer_train` (3D fullres, fold 0). Train Loss: 1.3490, Val Loss: 0.5406. Checkpoint: `model_final_checkpoint.model`.", table_cell)
        ],
        [
            Paragraph("Step 4", table_cell_bold),
            Paragraph("Model Inference", table_cell),
            Paragraph("<font color='#2F855A'><b>SUCCESS</b></font>", table_cell),
            Paragraph("`nnFormer_predict` 3D sliding window tiled inference (22 tiles) across `patient003.nii.gz` with 3D Gaussian weighting.", table_cell)
        ],
        [
            Paragraph("Step 5", table_cell_bold),
            Paragraph("Metric Evaluation", table_cell),
            Paragraph("<font color='#2F855A'><b>SUCCESS</b></font>", table_cell),
            Paragraph("Calculated Dice Similarity Coefficients per class across 3D anatomical structures.", table_cell)
        ],
    ]
    pipeline_table = Table(pipeline_data, colWidths=[45, 105, 60, 294])
    pipeline_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7FAFC")]),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(pipeline_table)
    elements.append(Spacer(1, 10))

    # Evaluation Metrics Table
    elements.append(Paragraph("3. Segmentation Accuracy & Metric Evaluation Results", h1_style))
    elements.append(Paragraph("Target Test Volume: <code>patient003.nii.gz</code> | Volume Dimensions: <code>14 x 160 x 160</code> voxels", body_style))
    
    metric_data = [
        [Paragraph("<b>Anatomical Region / Class</b>", table_header), Paragraph("<b>Label ID</b>", table_header), Paragraph("<b>Dice Similarity Coefficient (DSC)</b>", table_header), Paragraph("<b>Accuracy Percentage</b>", table_header)],
        [Paragraph("Right Ventricle (RV)", table_cell_bold), Paragraph("Class 1", table_cell), Paragraph("0.0640", table_cell), Paragraph("6.40%", table_cell)],
        [Paragraph("Myocardium (Myo)", table_cell_bold), Paragraph("Class 2", table_cell), Paragraph("0.2005", table_cell), Paragraph("20.05%", table_cell)],
        [Paragraph("Left Ventricle (LV)", table_cell_bold), Paragraph("Class 3", table_cell), Paragraph("0.0393", table_cell), Paragraph("3.93%", table_cell)],
        [Paragraph("<b>Global Foreground Mean</b>", table_cell_bold), Paragraph("<b>Classes 1–3</b>", table_cell_bold), Paragraph("<b>0.1013</b>", table_cell_bold), Paragraph("<b>10.13%</b>", table_cell_bold)],
    ]
    metric_table = Table(metric_data, colWidths=[160, 80, 144, 120])
    metric_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), secondary_color),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, colors.HexColor("#F7FAFC")]),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#EDF2F7")),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(metric_table)
    elements.append(Spacer(1, 10))

    # Technical Adaptations & Resolved Issues
    elements.append(Paragraph("4. Technical Adaptations & Compatibility Fixes", h1_style))
    elements.append(Paragraph("• <b>Windows Pickling Serialization Fix (PicklingError: Can't pickle lambda):</b> Replaced all anonymous <code>lambda</code> functions in <code>nd_softmax.py</code>, <code>neural_network.py</code>, <code>generic_UNet.py</code>, <code>nnFormer_acdc.py</code>, <code>nnFormer_synapse.py</code>, and <code>nnFormer_tumor.py</code> with top-level named functions (<code>softmax_helper</code>, <code>default_identity</code>, <code>identity_op</code>).", bullet_style))
    elements.append(Paragraph("• <b>Cross-Platform Path Resolution:</b> Replaced Unix hardcoded slash splitting in dataset conversion and checkpoint model restoration modules with Python <code>os.path</code> functions.", bullet_style))
    elements.append(Paragraph("• <b>Python 3 Generator Syntax:</b> Converted legacy <code>.next()</code> calls to Python 3 <code>next(...)</code> built-ins.", bullet_style))
    elements.append(Paragraph("• <b>Data Augmentation Windows Memory Paging Lock (WinError 1455):</b> Enforced single-threaded fallback (<code>nnFormer_n_proc_DA=1</code>) to prevent worker process memory paging locks on Windows.", bullet_style))

    elements.append(Spacer(1, 15))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E0"), spaceAfter=10))
    elements.append(Paragraph("<font color='#718096'>Report Generated Automatically after Clean End-to-End Pipeline Verification (Exit Code 0).</font>", ParagraphStyle('FooterNote', parent=styles['Normal'], fontSize=8, leading=10, textColor=colors.HexColor("#718096"))))

    doc.build(elements, canvasmaker=NumberedCanvas)
    print(f"PDF successfully created: {filename}")

if __name__ == "__main__":
    out_pdf = r"c:\Users\shara\Desktop\nnFormer\nnFormer_Execution_Report.pdf"
    create_pdf(out_pdf)
