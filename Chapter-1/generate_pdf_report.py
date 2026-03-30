from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import re
import os

def parse_markdown_to_pdf():
    """Parse the markdown file and generate a PDF report."""
    
    # Read the markdown file
    with open('Team_Experience_Analysis_Report.md', 'r', encoding='utf-8') as f:
        md_content = f.read()
    
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
        fontSize=22,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=13,
        textColor=colors.HexColor('#555555'),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica-Oblique'
    )
    
    heading1_style = ParagraphStyle(
        'CustomHeading1',
        parent=styles['Heading1'],
        fontSize=14,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=10,
        spaceBefore=10,
        fontName='Helvetica-Bold'
    )
    
    heading2_style = ParagraphStyle(
        'CustomHeading2',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=8,
        spaceBefore=8,
        fontName='Helvetica-Bold'
    )
    
    heading3_style = ParagraphStyle(
        'CustomHeading3',
        parent=styles['Heading3'],
        fontSize=11,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=6,
        spaceBefore=6,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#333333'),
        spaceAfter=8,
        alignment=TA_JUSTIFY,
        leading=12
    )
    
    bullet_style = ParagraphStyle(
        'CustomBullet',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#333333'),
        spaceAfter=6,
        leftIndent=20,
        leading=12
    )
    
    # Process markdown content line by line
    lines = md_content.split('\n')
    i = 0
    in_table = False
    table_data = []
    skip_until_section = False
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Title (H1 at top)
        if i < 5 and line.startswith('# '):
            title_text = line.replace('# ', '').strip()
            elements.append(Spacer(1, 1*inch))
            elements.append(Paragraph(title_text, title_style))
            
        # Subtitle (H2 right after title)
        elif i < 5 and line.startswith('## '):
            subtitle_text = line.replace('## ', '').strip()
            elements.append(Paragraph(subtitle_text, subtitle_style))
            elements.append(Spacer(1, 0.5*inch))
            
        # Main section headings (H2)
        elif line.startswith('## ') and not line.startswith('###'):
            heading_text = line.replace('## ', '').strip()
            if i > 10:  # Don't page break early sections
                elements.append(PageBreak())
            elements.append(Paragraph(heading_text, heading1_style))
            
        # Subsection headings (H3)
        elif line.startswith('### '):
            heading_text = line.replace('### ', '').strip()
            elements.append(Paragraph(heading_text, heading2_style))
            
        # Bold list items or bullet points
        elif line.startswith('- ') or line.startswith('* '):
            bullet_text = line[2:].strip()
            bullet_text = format_text(bullet_text)
            elements.append(Paragraph(f'• {bullet_text}', bullet_style))
            
        # Numbered lists
        elif re.match(r'^\d+\.', line):
            list_text = re.sub(r'^\d+\.\s*', '', line).strip()
            list_text = format_text(list_text)
            elements.append(Paragraph(list_text, body_style))
            
        # Tables
        elif line.startswith('|') and not in_table:
            in_table = True
            table_data = []
            # Process table header
            cols = [cell.strip() for cell in line.split('|')[1:-1]]
            table_data.append(cols)
            
        elif line.startswith('|') and in_table:
            # Check if it's a separator line
            if all(cell.strip().replace('-', '').replace(':', '') == '' for cell in line.split('|')[1:-1]):
                i += 1
                continue
            # Add table row
            cols = [cell.strip() for cell in line.split('|')[1:-1]]
            table_data.append(cols)
            
        elif in_table and not line.startswith('|'):
            # End of table, create it
            in_table = False
            if table_data:
                col_count = len(table_data[0])
                col_width = 6.5 * inch / col_count
                table = Table(table_data, colWidths=[col_width] * col_count)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', 8),
                    ('FONT', (0, 1), (-1, -1), 'Helvetica', 7),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ]))
                elements.append(table)
                elements.append(Spacer(1, 0.15*inch))
                table_data = []
            # Continue processing current line
            continue
            
        # Horizontal rules
        elif line.startswith('---') or line.startswith('***'):
            elements.append(Spacer(1, 0.1*inch))
            
        # Bold metadata lines
        elif line.startswith('**') and line.endswith('**'):
            text = line.replace('**', '').strip()
            elements.append(Paragraph(f'<b>{text}</b>', body_style))
            
        # Image references
        elif line.startswith('!['):
            # Extract image path from markdown: ![alt text](image.png)
            match = re.match(r'!\[([^\]]*)\]\(([^\)]+)\)', line)
            if match:
                alt_text = match.group(1)
                image_path = match.group(2)
                
                # Check if image file exists
                if os.path.exists(image_path):
                    try:
                        # Add image caption
                        if alt_text:
                            elements.append(Spacer(1, 0.1*inch))
                            caption_style = ParagraphStyle(
                                'Caption',
                                parent=body_style,
                                fontSize=9,
                                textColor=colors.HexColor('#555555'),
                                alignment=TA_CENTER,
                                fontName='Helvetica-Oblique'
                            )
                            elements.append(Paragraph(f'<b>{alt_text}</b>', caption_style))
                        
                        # Add the image - scale to fit width while maintaining aspect ratio
                        img = Image(image_path)
                        img._restrictSize(5.5*inch, 4*inch)  # Max width 5.5", max height 4"
                        elements.append(img)
                        elements.append(Spacer(1, 0.2*inch))
                    except Exception as e:
                        print(f"Warning: Could not load image {image_path}: {e}")
            
        # Regular paragraphs
        elif line and not line.startswith('#') and not line.startswith('|'):
            text = format_text(line)
            if text:
                elements.append(Paragraph(text, body_style))
        
        i += 1
    
    # Build PDF
    doc.build(elements)
    print(f"✓ PDF Report generated: {pdf_file}")

def format_text(text):
    """Format markdown text to ReportLab HTML-like markup."""
    # Bold text
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    # Italic text
    text = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', text)
    # Inline code (treat as bold)
    text = re.sub(r'`([^`]+)`', r'<b>\1</b>', text)
    # Remove markdown links but keep text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    return text

if __name__ == "__main__":
    parse_markdown_to_pdf()
