"""Summarize kriging diagnostics for the dashboard and the technical PDF."""

import io
from xml.sax.saxutils import escape

import numpy as np
from reportlab.graphics.shapes import Circle, Drawing, Line, PolyLine, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

NAVY = colors.HexColor("#10283b")
TEAL = colors.HexColor("#087e79")
ORANGE = colors.HexColor("#b95033")
GRAY = colors.HexColor("#526778")
RULE = colors.HexColor("#dce5eb")


def findings(summary):
    variogram = summary["variogram"]
    validation = summary["validation"]
    lag = np.asarray(variogram["lag"])
    gamma = np.asarray(variogram["gamma"])
    fitted = np.interp(lag, variogram["distance"], variogram["fitted"])
    residual = np.sqrt(np.mean((gamma - fitted) ** 2)) if len(lag) else None
    error = np.subtract(validation["predicted"], validation["observed"])
    bias = validation["mbe"]
    r2 = validation["r2"]
    if r2 is None:
        agreement = "R² is undefined because the observed values have no variation."
    elif r2 < 0:
        agreement = "The held-out predictions have more squared error than a baseline using the mean of the observed values."
    elif r2 == 0:
        agreement = "The held-out predictions match the observed-mean baseline in squared error."
    else:
        agreement = "The held-out predictions have less squared error than a baseline using the mean of the observed values."
    trend = "overpredicts" if bias > 0 else "underpredicts"
    bias_note = f"The model {trend} on average." if bias else "Average signed error is zero."
    return [
        dict(title="Model validation",
             summary=f"R² = {r2:.4g}." if r2 is not None else "R² is undefined.",
             conclusion=agreement),
        dict(title="Cross-validation",
             summary=f"Five-fold validation held out each of the {len(error)} samples once. "
                     f"RMSE was {validation['rmse']:.4g} and MAE was {validation['mae']:.4g} in source units.",
             conclusion="These errors describe held-out samples under random folds. "
                        "Performance in unsampled areas was not tested independently."),
        dict(title="Experimental variogram",
             summary=f"The {summary['model']} variogram used {summary['variogram_fit']} fitting. "
                     + (f"{len(lag)} populated lag bins span {lag.min():,.0f}–{lag.max():,.0f} m. "
                        f"The RMSE between experimental and modeled semivariance at these bins was {residual:.4g}."
                        if len(lag) else "No populated lag bins were available for comparison."),
             conclusion="The curve summarizes spatial dependence in the supplied samples. "
                        "Use cross-validation to assess prediction error."),
        dict(title="Observed vs. predicted",
             summary=f"{np.count_nonzero(error > 0)} predictions were above observed values, "
                     f"{np.count_nonzero(error < 0)} were below, and {np.count_nonzero(error == 0)} matched exactly. "
                     f"Mean signed error (predicted minus observed) was {bias:.4g}.",
             conclusion=bias_note + " Points farther from the 1:1 line have larger prediction errors."),
    ]


def plot(points, line, labels, equal_axes=False):
    drawing = Drawing(468, 161)
    left, bottom, width, height = 55, 31, 394, 109
    xy = np.asarray([*points, *line])
    low = xy.min(axis=0)
    high = xy.max(axis=0)
    if equal_axes:
        low[:] = low.min()
        high[:] = high.max()
    else:
        low[:] = 0
    span = high - low
    pad = np.where(span > 0, span * .04, np.maximum(abs(high) * .04, .01))
    if equal_axes:
        low -= pad
    high += pad
    span = high - low

    def position(x, y):
        return left + (x - low[0]) / span[0] * width, bottom + (y - low[1]) / span[1] * height

    for step in np.linspace(0, 1, 5):
        x, y = left + step * width, bottom + step * height
        drawing.add(Line(left, y, left + width, y, strokeColor=RULE, strokeWidth=.5))
        drawing.add(String(x, bottom - 12, f"{low[0] + step * span[0]:.4g}", textAnchor="middle", fontSize=8, fillColor=GRAY))
        drawing.add(String(left - 5, y - 2, f"{low[1] + step * span[1]:.4g}", textAnchor="end", fontSize=8, fillColor=GRAY))
    drawing.add(Line(left, bottom, left + width, bottom, strokeColor=GRAY, strokeWidth=.6))
    drawing.add(Line(left, bottom, left, bottom + height, strokeColor=GRAY, strokeWidth=.6))
    drawing.add(String(8, 149, labels[1], fontSize=8, fillColor=GRAY))
    drawing.add(String(left + width / 2, 3, labels[0], textAnchor="middle", fontSize=8, fillColor=GRAY))
    drawing.add(PolyLine([coordinate for point in line for coordinate in position(*point)], strokeColor=ORANGE, strokeWidth=1.2))
    for x, y in points:
        drawing.add(Circle(*position(x, y), 2.15, fillColor=TEAL, strokeColor=None))
    return drawing


def technical_pdf(summary, info, columns, rows, metrics):
    output = io.BytesIO()
    title = "Soil Fertility Mapping Technical Report"
    doc = SimpleDocTemplate(output, pagesize=A4, title=title, author="Soil Fertility Map Generator",
                            leftMargin=56, rightMargin=56, topMargin=59, bottomMargin=53)
    styles = getSampleStyleSheet()
    heading = ParagraphStyle("ReportHeading", parent=styles["Heading2"], textColor=NAVY,
                             fontSize=13, spaceBefore=11, spaceAfter=6)
    body = ParagraphStyle("ReportBody", parent=styles["BodyText"], fontSize=9.5, leading=13, textColor=NAVY)
    caption = ParagraphStyle("Caption", parent=body, fontSize=8.5, leading=11, textColor=GRAY)

    def table(headers, values, widths):
        result = Table([[Paragraph(f"<b>{escape(str(cell))}</b>", body) for cell in headers],
                        *[[Paragraph(escape(str(cell)), body) for cell in row] for row in values]],
                       colWidths=widths, repeatRows=1, hAlign="LEFT")
        result.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), RULE),
                                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                    ("LINEBELOW", (0, 0), (-1, -1), .35, RULE),
                                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
        return result

    def finding(item):
        return [Paragraph(escape(item["summary"]), body),
                Paragraph("<b>Conclusion.</b> " + escape(item["conclusion"]), body)]

    def page_frame(canvas, page):
        canvas.saveState()
        width, height = A4
        canvas.setFillColor(TEAL)
        canvas.rect(56, height - 43, 31, 3, fill=1, stroke=0)
        canvas.setFillColor(NAVY)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(95, height - 44, "SOIL FERTILITY MAP GENERATOR")
        canvas.setStrokeColor(RULE)
        canvas.line(56, 45, width - 56, 45)
        canvas.setFont("Helvetica", 8)
        canvas.drawString(56, 32, "Geostatistical mapping | Technical report")
        canvas.drawRightString(width - 56, 32, f"Page {page.page}")
        canvas.restoreState()

    model, cv, variogram, observed = summary["findings"]
    data = summary["variogram"]
    val = summary["validation"]
    low, high = min(*val["observed"], *val["predicted"]), max(*val["observed"], *val["predicted"])
    report = [
        Paragraph(title, styles["Title"]),
        Paragraph("1. Study area and method", heading),
        table(["Item", "Run details"], info, [146, 321]),
        Paragraph("2. Classification and mapped area", heading),
        table(columns, rows, [112, 157, 98, 100]),
        Paragraph("3. Model validation", heading),
        table(["Metric", "Value"], metrics, [285, 182]), Spacer(1, 5),
        *finding(model),
        Paragraph("4. Five-fold cross-validation", heading), *finding(cv),
        PageBreak(),
        Paragraph("5. Experimental variogram", heading),
        plot(list(zip(data["lag"], data["gamma"])), list(zip(data["distance"], data["fitted"])), ("Distance (m)", "Semivariance")),
        Paragraph("Figure 1. Experimental lag bins (dots) and model fit (line).", caption), Spacer(1, 5), *finding(variogram),
        Paragraph("6. Observed vs. predicted", heading),
        plot(list(zip(val["observed"], val["predicted"])), [(low, low), (high, high)], ("Observed", "Predicted"), equal_axes=True),
        Paragraph("Figure 2. Out-of-fold predictions (dots) against the 1:1 reference line.", caption), Spacer(1, 5), *finding(observed),
    ]
    doc.build(report, onFirstPage=page_frame, onLaterPages=page_frame)
    return output.getvalue()
