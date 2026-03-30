from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import os
import re

# Create PDF
pdf_file = "Team_Experience_Analysis_Report.pdf"
doc = SimpleDocTemplate(pdf_file, pagesize=letter,
                        rightMargin=72, leftMargin=72,
                        topMargin=72, bottomMargin=72)

# Container for the 'Flowable' objects
elements = []

# Define styles
styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    'CustomTitle',
    parent=styles['Heading1'],
    fontSize=24,
    textColor=colors.HexColor('#1a1a1a'),
    spaceAfter=30,
    alignment=TA_CENTER,
    fontName='Helvetica-Bold'
)

subtitle_style = ParagraphStyle(
    'CustomSubtitle',
    parent=styles['Normal'],
    fontSize=14,
    textColor=colors.HexColor('#555555'),
    spaceAfter=30,
    alignment=TA_CENTER,
    fontName='Helvetica-Oblique'
)

heading1_style = ParagraphStyle(
    'CustomHeading1',
    parent=styles['Heading1'],
    fontSize=16,
    textColor=colors.HexColor('#2c3e50'),
    spaceAfter=12,
    spaceBefore=12,
    fontName='Helvetica-Bold'
)

heading2_style = ParagraphStyle(
    'CustomHeading2',
    parent=styles['Heading2'],
    fontSize=13,
    textColor=colors.HexColor('#34495e'),
    spaceAfter=10,
    spaceBefore=10,
    fontName='Helvetica-Bold'
)

body_style = ParagraphStyle(
    'CustomBody',
    parent=styles['Normal'],
    fontSize=10,
    textColor=colors.HexColor('#333333'),
    spaceAfter=12,
    alignment=TA_JUSTIFY,
    leading=14
)

# Title Page
elements.append(Spacer(1, 1.5*inch))
title = Paragraph("Team Experience Analysis Report", title_style)
elements.append(title)
elements.append(Spacer(1, 0.3*inch))

subtitle = Paragraph("Understanding Team Conditions That Drive Performance, Learning, and Growth", subtitle_style)
elements.append(subtitle)
elements.append(Spacer(1, 1*inch))

# Metadata
meta_data = [
    ["Date:", "January 14, 2026"],
    ["Dataset:", "210 student survey responses from IS 415"],
    ["Analysis Type:", "Team Experience & Performance Analysis"]
]
meta_table = Table(meta_data, colWidths=[2*inch, 4*inch])
meta_table.setStyle(TableStyle([
    ('FONT', (0, 0), (-1, -1), 'Helvetica', 10),
    ('FONT', (0, 0), (0, -1), 'Helvetica-Bold', 10),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
]))
elements.append(meta_table)
elements.append(PageBreak())

# Executive Summary
elements.append(Paragraph("Executive Summary", heading1_style))
elements.append(Paragraph(
    "This report analyzes survey data from 210 students reflecting on their team experiences from the previous semester. "
    "Our goal was to identify which team conditions most strongly predict positive outcomes in performance, learning, and growth—"
    "and to translate these findings into actionable recommendations for improving future team formation and support strategies.",
    body_style
))
elements.append(Spacer(1, 0.2*inch))

# Key Findings
elements.append(Paragraph("Key Findings", heading2_style))
findings = [
    "<b>1. Psychological Safety is the strongest predictor of team performance</b> (r = 0.47), surpassing even team cohesion "
    "and social identity. Teams where members felt safe to take risks and make mistakes consistently delivered higher-quality work.",
    
    "<b>2. High-performing teams want to work together again.</b> 87% of teams rating their performance as 'excellent' (10/10) "
    "expressed high willingness to collaborate in the future, compared to only 43% of low-performing teams.",
    
    "<b>3. Psychological Safety drives learning and confidence.</b> Teams with high psychological safety reported 8% higher "
    "learning gains and 23% higher self-efficacy compared to teams with low psychological safety.",
    
    "<b>4. Social Identity and Team Cohesion are the strongest predictors of improvement over time</b> (r = 0.69 and r = 0.66). "
    "Teams that developed a strong 'we' identity and interpersonal bonds showed the greatest growth throughout the semester."
]
for finding in findings:
    elements.append(Paragraph(finding, body_style))
    elements.append(Spacer(1, 0.1*inch))

elements.append(Spacer(1, 0.2*inch))

# Strategic Implications
elements.append(Paragraph("Strategic Implications", heading2_style))
elements.append(Paragraph(
    "These findings suggest that <b>team formation strategies should prioritize creating conditions for psychological safety "
    "and social bonding early in the semester</b>, rather than focusing solely on skill matching or workload distribution. "
    "Specific recommendations include structured team-building exercises, explicit norms around mistake tolerance, and early "
    "interventions for teams showing low cohesion.",
    body_style
))

elements.append(PageBreak())

# Section 1: Introduction
elements.append(Paragraph("1. Introduction", heading1_style))
elements.append(Paragraph("Background and Purpose", heading2_style))
elements.append(Paragraph(
    "Faculty use team-based projects intentionally to develop collaboration skills, enhance learning through peer interaction, "
    "and prepare students for professional environments. However, not all team experiences are equally valuable. "
    "This analysis seeks to understand: Which team conditions lead to better performance and output quality? "
    "How do these conditions affect learning gains and confidence development? What early indicators predict teams that will "
    "improve versus those that will stagnate?",
    body_style
))
elements.append(Spacer(1, 0.15*inch))

elements.append(Paragraph("Data and Methodology", heading2_style))
elements.append(Paragraph(
    "<b>Dataset:</b> 210 anonymized student surveys from IS 415, reflecting on team experiences from the previous semester.",
    body_style
))
elements.append(Paragraph(
    "<b>Key Variables Analyzed:</b> Team Cohesion (1-5), Social Identity & Belonging (6-10), Psychological Safety (1-5), "
    "Self-Efficacy (1-5), Performance (6-10), Learning (6-10), Growth (1-5), and Willingness to Work Together (1-5).",
    body_style
))
elements.append(Paragraph(
    "<b>Analytical Approach:</b> Correlation analysis, group comparisons (high vs. low performers), and visual exploration.",
    body_style
))

elements.append(PageBreak())

# Analysis 1
elements.append(Paragraph("2. Analysis 1: Which Team Conditions Most Strongly Predict Performance?", heading1_style))
elements.append(Paragraph("Research Question", heading2_style))
elements.append(Paragraph(
    "Which team experience factors—cohesion, psychological safety, social identity, confidence, or willingness to collaborate—"
    "appear most strongly associated with perceived team performance and output quality?",
    body_style
))
elements.append(Spacer(1, 0.15*inch))

elements.append(Paragraph("Findings", heading2_style))
corr_data = [
    ["Factor", "Correlation (r)", "Significance"],
    ["Psychological Safety", "0.471", "p < 0.001 ***"],
    ["Social Identity & Belonging", "0.446", "p < 0.001 ***"],
    ["Willingness to Work Together", "0.428", "p < 0.001 ***"],
    ["Team Cohesion", "0.380", "p < 0.001 ***"],
    ["Self-Efficacy", "0.319", "p < 0.001 ***"]
]
corr_table = Table(corr_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
corr_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 10),
    ('FONT', (0, 1), (-1, -1), 'Helvetica', 9),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black)
]))
elements.append(corr_table)
elements.append(Spacer(1, 0.15*inch))

# Add image if it exists
if os.path.exists('Fig1_Performance_Correlations.png'):
    img = Image('Fig1_Performance_Correlations.png', width=6*inch, height=4*inch)
    elements.append(img)
    elements.append(Paragraph("<i>Figure 1: Team experience factors vs. performance</i>", body_style))
    elements.append(Spacer(1, 0.15*inch))

elements.append(Paragraph("Interpretation", heading2_style))
elements.append(Paragraph(
    "<b>Psychological Safety emerges as the single strongest predictor of team performance</b> (r = 0.47), followed closely "
    "by Social Identity (r = 0.45). This pattern suggests that performance is not just about skill—it's about environment. "
    "Teams where members felt safe to take interpersonal risks consistently produced higher-quality work. "
    "Importantly, confidence alone is insufficient: Self-efficacy shows the weakest correlation with performance (r = 0.32).",
    body_style
))
elements.append(Spacer(1, 0.1*inch))

elements.append(Paragraph("<b>Key Takeaway:</b> Creating psychologically safe team environments should be a priority "
                         "if performance quality is the goal.", body_style))

elements.append(PageBreak())

# Analysis 2
elements.append(Paragraph("3. Analysis 2: Do High-Performing Teams Want to Work Together Again?", heading1_style))
elements.append(Paragraph("Research Question", heading2_style))
elements.append(Paragraph(
    "Are teams that perform well also the teams students would choose to work with again? "
    "Is there alignment between performance quality and team satisfaction?",
    body_style
))
elements.append(Spacer(1, 0.15*inch))

elements.append(Paragraph("Findings", heading2_style))
elements.append(Paragraph("<b>Correlation:</b> r = 0.428 (p < 0.001)", body_style))
elements.append(Spacer(1, 0.1*inch))

perf_will_data = [
    ["Performance Level", "% High Willingness", "% Medium Willingness", "% Low Willingness"],
    ["High (10/10)", "87.5%", "11.6%", "0.9%"],
    ["Medium (9/10)", "50.0%", "32.4%", "17.6%"],
    ["Low (≤8/10)", "43.3%", "30.0%", "26.7%"]
]
perf_will_table = Table(perf_will_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
perf_will_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 9),
    ('FONT', (0, 1), (-1, -1), 'Helvetica', 9),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black)
]))
elements.append(perf_will_table)
elements.append(Spacer(1, 0.15*inch))

if os.path.exists('Fig2_Performance_vs_Willingness.png'):
    img = Image('Fig2_Performance_vs_Willingness.png', width=6*inch, height=3*inch)
    elements.append(img)
    elements.append(Paragraph("<i>Figure 2: Performance vs. willingness to collaborate again</i>", body_style))
    elements.append(Spacer(1, 0.15*inch))

elements.append(Paragraph("Interpretation", heading2_style))
elements.append(Paragraph(
    "<b>There is strong alignment between performance quality and desire to work together again.</b> "
    "Teams that produce excellent work (10/10) are overwhelmingly positive about future collaboration (87.5%), "
    "while low performers are much more ambivalent. This suggests that performance and satisfaction are not in tension—"
    "they reinforce each other.",
    body_style
))

elements.append(PageBreak())

# Analysis 3
elements.append(Paragraph("4. Analysis 3: How Does Psychological Safety Relate to Learning?", heading1_style))
elements.append(Paragraph("Research Question", heading2_style))
elements.append(Paragraph(
    "How does psychological safety relate to learning gains and confidence development? "
    "Do teams where mistakes are treated as part of learning show stronger outcomes?",
    body_style
))
elements.append(Spacer(1, 0.15*inch))

elements.append(Paragraph("Findings", heading2_style))
elements.append(Paragraph(
    "<b>Correlations:</b> Psychological Safety → Learning: r = 0.245 (p < 0.001) | "
    "Psychological Safety → Self-Efficacy: r = 0.326 (p < 0.001)",
    body_style
))
elements.append(Spacer(1, 0.1*inch))

ps_data = [
    ["PS Level", "Avg Learning", "Avg Self-Efficacy", "Avg Performance", "n"],
    ["Low (1-3)", "8.80 (±1.24)", "3.65 (±0.84)", "8.50 (±1.00)", "20"],
    ["Medium (4)", "9.39 (±0.64)", "4.12 (±0.81)", "9.00 (±0.86)", "36"],
    ["High (5)", "9.50 (±0.79)", "4.45 (±0.65)", "9.53 (±0.73)", "154"]
]
ps_table = Table(ps_data, colWidths=[1.2*inch, 1.3*inch, 1.5*inch, 1.5*inch, 0.7*inch])
ps_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 9),
    ('FONT', (0, 1), (-1, -1), 'Helvetica', 9),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black)
]))
elements.append(ps_table)
elements.append(Spacer(1, 0.15*inch))

if os.path.exists('Fig3_PsychSafety_Learning.png'):
    img = Image('Fig3_PsychSafety_Learning.png', width=6*inch, height=3*inch)
    elements.append(img)
    elements.append(Paragraph("<i>Figure 3: Psychological safety relationships with learning and confidence</i>", body_style))
    elements.append(Spacer(1, 0.15*inch))

elements.append(Paragraph("Interpretation", heading2_style))
elements.append(Paragraph(
    "<b>Teams with high psychological safety not only perform better—they also learn more and build greater confidence.</b> "
    "High PS teams reported learning scores 8% higher than low PS teams (9.50 vs. 8.80), and confidence development was even "
    "more sensitive, showing 22% higher self-efficacy (4.45 vs. 3.65). This demonstrates that psychological safety is not just "
    "about comfort—it's about creating conditions for learning and skill development.",
    body_style
))

elements.append(PageBreak())

# Analysis 4
elements.append(Paragraph("5. Analysis 4: What Team Conditions Predict Growth Over Time?", heading1_style))
elements.append(Paragraph("Research Question", heading2_style))
elements.append(Paragraph(
    "What differentiates teams that improved over the semester from those that stagnated? "
    "Which early team conditions seem to predict improvement later on?",
    body_style
))
elements.append(Spacer(1, 0.15*inch))

elements.append(Paragraph("Findings", heading2_style))
growth_corr_data = [
    ["Factor", "Correlation (r)", "Significance"],
    ["Social Identity & Belonging", "0.693", "p < 0.001 ***"],
    ["Team Cohesion", "0.659", "p < 0.001 ***"],
    ["Self-Efficacy", "0.487", "p < 0.001 ***"],
    ["Psychological Safety", "0.331", "p < 0.001 ***"]
]
growth_corr_table = Table(growth_corr_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
growth_corr_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 10),
    ('FONT', (0, 1), (-1, -1), 'Helvetica', 9),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black)
]))
elements.append(growth_corr_table)
elements.append(Spacer(1, 0.15*inch))

if os.path.exists('Fig4_Growth_Predictors.png'):
    img = Image('Fig4_Growth_Predictors.png', width=6*inch, height=3*inch)
    elements.append(img)
    elements.append(Paragraph("<i>Figure 4: Team growth predictors and outcomes by growth level</i>", body_style))
    elements.append(Spacer(1, 0.15*inch))

elements.append(Paragraph("Interpretation", heading2_style))
elements.append(Paragraph(
    "<b>Social Identity and Team Cohesion are the strongest predictors of team improvement over time</b> (r = 0.69 and r = 0.66). "
    "Teams that developed strong 'we-ness' early in the semester showed the greatest capacity to adapt, learn, and improve. "
    "Interestingly, while Psychological Safety predicts current performance strongly (r = 0.47), it shows weaker correlation "
    "with improvement over time (r = 0.33), suggesting that PS enables consistent performance while cohesion drives adaptation.",
    body_style
))
elements.append(Spacer(1, 0.1*inch))
elements.append(Paragraph(
    "<b>Key Takeaway:</b> If the goal is continuous improvement, prioritize team-building activities that foster social identity "
    "and cohesion early in the semester.",
    body_style
))

elements.append(PageBreak())

# Comparative Analysis
elements.append(Paragraph("6. Comparative Analysis: High vs. Low Performing Teams", heading1_style))
comp_data = [
    ["Metric", "High (n=112)", "Low (n=30)", "Difference"],
    ["Team Cohesion", "4.33", "3.40", "+0.93"],
    ["Social Identity", "9.63", "8.63", "+1.00"],
    ["Psychological Safety", "4.75", "3.60", "+1.15"],
    ["Self-Efficacy", "4.50", "3.77", "+0.73"],
    ["Willingness (Future)", "4.72", "3.85", "+0.87"],
    ["Learning", "9.52", "8.93", "+0.59"],
    ["Growth", "4.77", "3.60", "+1.17"]
]
comp_table = Table(comp_data, colWidths=[2*inch, 1.3*inch, 1.3*inch, 1.3*inch])
comp_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (0, -1), 'LEFT'),
    ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
    ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 10),
    ('FONT', (0, 1), (-1, -1), 'Helvetica', 9),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 1, colors.black)
]))
elements.append(comp_table)
elements.append(Spacer(1, 0.15*inch))

if os.path.exists('Fig6_High_vs_Low_Performers.png'):
    img = Image('Fig6_High_vs_Low_Performers.png', width=6*inch, height=3*inch)
    elements.append(img)
    elements.append(Paragraph("<i>Figure 6: Comparison of high vs. low performing teams</i>", body_style))
    elements.append(Spacer(1, 0.15*inch))

elements.append(Paragraph(
    "High-performing teams show substantially higher scores across all dimensions, with the largest differences in "
    "Psychological Safety (+1.15), Growth (+1.17), and Social Identity (+1.00). Importantly, the differences are substantial "
    "but not extreme, suggesting that <b>small improvements in team conditions could shift teams from 'adequate' to 'excellent.'</b>",
    body_style
))

elements.append(PageBreak())

# Recommendations
elements.append(Paragraph("7. Recommendations and Implications for Decision-Makers", heading1_style))
elements.append(Paragraph(
    "Based on the findings, we offer the following evidence-based recommendations for improving future team experiences:",
    body_style
))
elements.append(Spacer(1, 0.15*inch))

recommendations = [
    ("<b>1. Prioritize Psychological Safety from Day One</b>", 
     "Establish explicit norms early (e.g., 'mistakes are learning opportunities'), model vulnerability, and check in on PS regularly."),
    
    ("<b>2. Invest in Team-Building Activities That Foster Social Identity</b>", 
     "Use structured icebreakers with depth, create team identity artifacts, and allocate time for non-task bonding."),
    
    ("<b>3. Intervene Early with Low-Cohesion Teams</b>", 
     "Administer brief surveys after Week 2-3, offer targeted support for struggling teams, and consider team reconfiguration as a last resort."),
    
    ("<b>4. Frame Teamwork as a Learning Opportunity, Not Just a Deliverable</b>", 
     "Emphasize process over outcomes, celebrate productive failures, and assess individual contributions separately from team output."),
    
    ("<b>5. Design for Continuous Improvement, Not Just Initial Success</b>", 
     "Build in structured reflection points, normalize iteration, and provide scaffolding for adaptation.")
]

for title, desc in recommendations:
    elements.append(Paragraph(title, heading2_style))
    elements.append(Paragraph(desc, body_style))
    elements.append(Spacer(1, 0.1*inch))

elements.append(PageBreak())

# Conclusion
elements.append(Paragraph("8. Conclusion", heading1_style))
elements.append(Paragraph(
    "This analysis demonstrates that <b>team performance, learning, and growth are not random outcomes—they are systematically "
    "predicted by specific team conditions</b>, particularly psychological safety, social identity, and team cohesion.",
    body_style
))
elements.append(Spacer(1, 0.15*inch))
elements.append(Paragraph(
    "The most actionable insight is this: <b>small, intentional investments in team culture early in the semester can produce "
    "substantial returns in both work quality and student experience.</b> Creating norms that encourage risk-taking, fostering "
    "social bonds that build 'we-ness,' and identifying struggling teams early for targeted support are all evidence-based "
    "strategies that can improve outcomes.",
    body_style
))
elements.append(Spacer(1, 0.15*inch))
elements.append(Paragraph(
    "Importantly, these recommendations are not in tension with academic rigor or performance standards. To the contrary, "
    "our findings suggest that <b>teams with strong interpersonal dynamics produce better work, learn more, and develop greater "
    "confidence</b>—achieving both relational and task outcomes simultaneously.",
    body_style
))
elements.append(Spacer(1, 0.15*inch))
elements.append(Paragraph(
    "We encourage faculty and program administrators to consider these findings when designing team-based assignments, "
    "structuring team formation processes, and determining when and how to intervene with struggling teams.",
    body_style
))

elements.append(Spacer(1, 0.3*inch))
elements.append(Paragraph("<i>End of Report</i>", subtitle_style))

# Build PDF
doc.build(elements)
print(f"✓ PDF Report generated: {pdf_file}")
