from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font('helvetica', size=12)
pdf.multi_cell(w=0,text="""Q: How do I return a product?
A: You can return any product within 30 days with a receipt.

Q: How long does refund take?
A: Refunds are processed within 5-7 business days.

Q: How do I contact support?
A: Email us at support@shopease.com or call 1800-XXX-XXX.

Q: Can I exchange a product?
A: Yes, exchanges are allowed within 15 days of purchase.""")
pdf.output("custe.pdf")