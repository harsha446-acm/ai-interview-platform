from io import BytesIO
import tempfile
import os
from fpdf import FPDF
from datetime import datetime
import numpy as np

# Use non-interactive backend (no GUI needed on server)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── Chart Helpers ─────────────────────────────────────

def _chart_to_tempfile(fig) -> str:
    """Save a matplotlib figure to a temp PNG file and return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return tmp.name


def _make_radar_chart(scores: dict) -> str:
    """Create a radar/spider chart of overall skill scores."""
    categories = ["Content", "Keywords", "Depth", "Communication", "Confidence"]
    values = [
        scores.get("content_score", 0),
        scores.get("keyword_score", 0),
        scores.get("depth_score", 0),
        scores.get("communication_score", 0),
        scores.get("confidence_score", 0),
    ]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))
    ax.fill(angles, values_plot, color="#667eea", alpha=0.25)
    ax.plot(angles, values_plot, color="#667eea", linewidth=2, marker="o", markersize=6)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=7, color="gray")
    ax.set_title("Skills Breakdown", fontsize=13, fontweight="bold", pad=20, color="#333")
    ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    return _chart_to_tempfile(fig)


def _make_question_bar_chart(evaluations: list) -> str:
    """Create a bar chart showing per-question scores color-coded by round."""
    if not evaluations:
        return None

    labels = []
    scores = []
    colors = []
    for i, ev in enumerate(evaluations, 1):
        labels.append(f"Q{i}")
        scores.append(ev.get("scores", {}).get("overall_score", 0))
        colors.append("#667eea" if ev.get("round", "Technical") == "Technical" else "#f59e0b")

    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 0.7), 3.5))
    bars = ax.bar(labels, scores, color=colors, width=0.6, edgecolor="white", linewidth=0.5)

    # Add score labels on top of bars
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{score:.0f}", ha="center", va="bottom", fontsize=7, fontweight="bold", color="#333")

    ax.set_ylim(0, 110)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Question-wise Scores", fontsize=13, fontweight="bold", color="#333")
    ax.axhline(y=70, color="#22c55e", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axhline(y=40, color="#ef4444", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    tech_patch = mpatches.Patch(color="#667eea", label="Technical")
    hr_patch = mpatches.Patch(color="#f59e0b", label="HR")
    ax.legend(handles=[tech_patch, hr_patch], fontsize=8, loc="upper right")

    return _chart_to_tempfile(fig)


def _make_round_comparison_chart(tech_score: float, hr_score: float,
                                  tech_count: int, hr_count: int) -> str:
    """Create a grouped bar chart comparing Technical vs HR round performance."""
    fig, ax = plt.subplots(figsize=(4, 3.5))

    categories = ["Avg Score", "Questions"]
    tech_vals = [tech_score, tech_count]
    hr_vals = [hr_score, hr_count]

    x = np.arange(len(categories))
    width = 0.3

    bars1 = ax.bar(x - width / 2, tech_vals, width, label="Technical", color="#667eea", edgecolor="white")
    bars2 = ax.bar(x + width / 2, hr_vals, width, label="HR", color="#f59e0b", edgecolor="white")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_title("Round Comparison", fontsize=13, fontweight="bold", color="#333")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return _chart_to_tempfile(fig)


def _make_score_distribution_chart(evaluations: list) -> str:
    """Create a horizontal stacked bar showing score component breakdown per question."""
    if not evaluations or len(evaluations) < 2:
        return None

    labels = [f"Q{i}" for i in range(1, len(evaluations) + 1)]
    content = [e.get("scores", {}).get("content_score", 0) for e in evaluations]
    comm = [e.get("scores", {}).get("communication_score", 0) for e in evaluations]
    depth = [e.get("scores", {}).get("depth_score", 0) for e in evaluations]

    fig, ax = plt.subplots(figsize=(5.5, max(2.5, len(labels) * 0.35)))

    y = np.arange(len(labels))
    height = 0.5

    ax.barh(y, content, height, label="Content", color="#667eea", alpha=0.85)
    ax.barh(y, comm, height, left=content, label="Communication", color="#22c55e", alpha=0.85)
    ax.barh(y, depth, height, left=np.array(content) + np.array(comm), label="Depth", color="#f59e0b", alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Cumulative Score", fontsize=9)
    ax.set_title("Score Components per Question", fontsize=12, fontweight="bold", color="#333")
    ax.legend(fontsize=8, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return _chart_to_tempfile(fig)


# ── PDF Report Generator ─────────────────────────────

def generate_pdf_report(report: dict) -> bytes:
    """Generate a PDF performance report with embedded charts."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    chart_files = []  # track temp files for cleanup

    try:
        pdf.add_page()

        # ── Title ─────────────────────────────────────
        pdf.set_font("Helvetica", "B", 22)
        pdf.set_text_color(102, 126, 234)
        pdf.cell(0, 15, "Interview Performance Report", ln=True, align="C")
        pdf.ln(3)

        # Decorative line
        pdf.set_draw_color(102, 126, 234)
        pdf.set_line_width(0.8)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        # ── Candidate info ────────────────────────────
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(80, 80, 80)
        info_items = [
            ("Candidate", report.get("candidate_name", "N/A")),
            ("Role", report.get("job_role", "N/A")),
            ("Date", datetime.now().strftime("%B %d, %Y")),
            ("Questions", str(report.get("total_questions", 0))),
            ("Recommendation", report.get("recommendation", "N/A")),
        ]
        for label, value in info_items:
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(40, 7, f"{label}:")
            pdf.set_font("Helvetica", "", 11)
            pdf.set_text_color(30, 30, 30)
            pdf.cell(0, 7, value, ln=True)
        pdf.ln(5)

        # ── Overall Scores Table ──────────────────────
        scores = report.get("overall_scores", {})
        _section_header(pdf, "Overall Scores", (102, 126, 234))

        score_items = [
            ("Content Score (40%)", scores.get("content_score", 0)),
            ("Communication Score (30%)", scores.get("communication_score", 0)),
            ("Confidence Score (20%)", scores.get("confidence_score", 0)),
            ("Depth Score", scores.get("depth_score", 0)),
            ("Keyword Coverage", scores.get("keyword_score", 0)),
        ]

        # Table header
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(245, 245, 250)
        pdf.cell(95, 8, "  Metric", border=1, fill=True)
        pdf.cell(35, 8, "Score", border=1, align="C", fill=True)
        pdf.cell(60, 8, "Rating", border=1, align="C", fill=True, ln=True)

        pdf.set_font("Helvetica", "", 10)
        for label, val in score_items:
            pdf.set_text_color(30, 30, 30)
            pdf.cell(95, 7, f"  {label}", border=1)
            _score_cell(pdf, val, 35)
            rating = "Excellent" if val >= 80 else "Good" if val >= 60 else "Fair" if val >= 40 else "Needs Work"
            pdf.cell(60, 7, rating, border=1, align="C", ln=True)

        # Overall score row (bold)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_fill_color(102, 126, 234)
        pdf.set_text_color(255, 255, 255)
        overall = scores.get("overall_score", 0)
        pdf.cell(95, 9, "  OVERALL SCORE", border=1, fill=True)
        pdf.cell(35, 9, f"{overall:.1f}", border=1, align="C", fill=True)
        rating = "Excellent" if overall >= 80 else "Good" if overall >= 60 else "Fair" if overall >= 40 else "Needs Work"
        pdf.cell(60, 9, rating, border=1, align="C", fill=True, ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)

        # ── CHARTS PAGE ──────────────────────────────
        pdf.add_page()
        _section_header(pdf, "Performance Analytics", (102, 126, 234))
        pdf.ln(2)

        # Radar chart (left) + Round comparison (right)
        radar_file = _make_radar_chart(scores)
        chart_files.append(radar_file)
        chart_y = pdf.get_y()
        pdf.image(radar_file, x=10, y=chart_y, w=90)

        tech_score = report.get("technical_score", 0)
        hr_score = report.get("hr_score", 0)
        tech_count = report.get("technical_questions", 0)
        hr_count = report.get("hr_questions", 0)
        round_file = _make_round_comparison_chart(tech_score, hr_score, tech_count, hr_count)
        chart_files.append(round_file)
        pdf.image(round_file, x=105, y=chart_y, w=95)

        pdf.set_y(chart_y + 72)
        pdf.ln(5)

        # Question-wise bar chart (full width)
        evaluations = report.get("question_evaluations", [])
        q_bar_file = _make_question_bar_chart(evaluations)
        if q_bar_file:
            chart_files.append(q_bar_file)
            pdf.image(q_bar_file, x=10, w=190)
            pdf.ln(5)

        # Score components stacked chart
        dist_file = _make_score_distribution_chart(evaluations)
        if dist_file:
            chart_files.append(dist_file)
            if pdf.get_y() > 200:
                pdf.add_page()
            pdf.image(dist_file, x=20, w=170)
            pdf.ln(5)

        # ── Strengths ─────────────────────────────────
        pdf.add_page()
        _section_header(pdf, "Strengths", (34, 139, 34))
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(0, 0, 0)
        for s in report.get("strengths", []):
            pdf.cell(5)
            pdf.set_text_color(34, 139, 34)
            pdf.cell(5, 7, chr(10004))  # checkmark
            pdf.set_text_color(30, 30, 30)
            pdf.cell(0, 7, f" {s}", ln=True)
        pdf.ln(5)

        # ── Weaknesses ────────────────────────────────
        _section_header(pdf, "Areas for Improvement", (220, 20, 60))
        pdf.set_font("Helvetica", "", 11)
        for w in report.get("weaknesses", []):
            pdf.cell(5)
            pdf.set_text_color(220, 20, 60)
            pdf.cell(5, 7, chr(10008))  # X mark
            pdf.set_text_color(30, 30, 30)
            pdf.cell(0, 7, f" {w}", ln=True)
        pdf.ln(5)

        # ── Suggestions ───────────────────────────────
        _section_header(pdf, "Improvement Suggestions", (255, 165, 0))
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(30, 30, 30)
        for idx, s in enumerate(report.get("improvement_suggestions", []), 1):
            pdf.cell(5)
            pdf.cell(0, 7, f"{idx}. {s}", ln=True)
        pdf.ln(3)

        # ── Communication & Confidence ────────────────
        comm_fb = report.get("communication_feedback", "")
        conf_fb = report.get("confidence_analysis", "")
        if comm_fb or conf_fb:
            _section_header(pdf, "Detailed Feedback", (102, 126, 234))
            pdf.set_font("Helvetica", "", 11)
            pdf.set_text_color(30, 30, 30)
            if comm_fb:
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(0, 7, "Communication:", ln=True)
                pdf.set_font("Helvetica", "", 10)
                pdf.multi_cell(0, 6, f"  {comm_fb}")
                pdf.ln(2)
            if conf_fb:
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(0, 7, "Overall Assessment:", ln=True)
                pdf.set_font("Helvetica", "", 10)
                pdf.multi_cell(0, 6, f"  {conf_fb}")
            pdf.ln(5)

        # ── Question-wise Breakdown ───────────────────
        pdf.add_page()
        _section_header(pdf, "Question-wise Breakdown", (102, 126, 234))
        pdf.ln(3)

        for idx, qe in enumerate(evaluations, 1):
            if pdf.get_y() > 220:
                pdf.add_page()

            # Question header with colored badge
            round_type = qe.get("round", "Technical")
            badge_color = (102, 126, 234) if round_type == "Technical" else (245, 158, 11)
            pdf.set_fill_color(*badge_color)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(20, 6, f" {round_type}", fill=True)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(30, 30, 30)

            # Truncate long questions
            q_text = qe.get("question", "")
            if len(q_text) > 90:
                q_text = q_text[:87] + "..."
            pdf.cell(0, 6, f"  Q{idx}: {q_text}", ln=True)

            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(80, 80, 80)

            # Answer (truncated)
            ans = qe.get("answer", "N/A")
            if len(ans) > 250:
                ans = ans[:247] + "..."
            pdf.multi_cell(0, 5, f"Answer: {ans}")
            pdf.ln(1)

            # Inline score bar
            q_scores = qe.get("scores", {})
            q_overall = q_scores.get("overall_score", 0)
            _inline_score_bar(pdf, "Overall", q_overall)
            _inline_score_bar(pdf, "Content", q_scores.get("content_score", 0))
            _inline_score_bar(pdf, "Comm", q_scores.get("communication_score", 0))

            # Keywords
            matched = qe.get("keywords_matched", [])
            missed = qe.get("keywords_missed", [])
            if matched:
                pdf.set_text_color(34, 139, 34)
                pdf.set_font("Helvetica", "", 8)
                pdf.cell(0, 5, f"  Keywords hit: {', '.join(matched[:6])}", ln=True)
            if missed:
                pdf.set_text_color(220, 20, 60)
                pdf.set_font("Helvetica", "", 8)
                pdf.cell(0, 5, f"  Keywords missed: {', '.join(missed[:6])}", ln=True)

            # Feedback
            feedback = qe.get("feedback", "")
            if feedback:
                pdf.set_font("Helvetica", "I", 9)
                pdf.set_text_color(100, 100, 100)
                if len(feedback) > 200:
                    feedback = feedback[:197] + "..."
                pdf.multi_cell(0, 5, f"  Feedback: {feedback}")

            pdf.set_text_color(0, 0, 0)
            pdf.ln(4)

        # ── Footer on last page ───────────────────────
        pdf.set_y(-25)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(150, 150, 150)
        pdf.cell(0, 5, f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')} | AI Interview Platform", align="C")

        return bytes(pdf.output())

    finally:
        # Cleanup temp chart image files
        for f in chart_files:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except OSError:
                    pass


# ── Layout Helpers ────────────────────────────────────

def _section_header(pdf: FPDF, title: str, color: tuple):
    """Render a styled section header with underline."""
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(*color)
    pdf.cell(0, 10, title, ln=True)
    pdf.set_draw_color(*color)
    pdf.set_line_width(0.5)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)


def _score_cell(pdf: FPDF, val: float, width: int):
    """Render a score cell with color coding."""
    if val >= 70:
        pdf.set_text_color(34, 139, 34)
    elif val >= 40:
        pdf.set_text_color(255, 140, 0)
    else:
        pdf.set_text_color(220, 20, 60)
    pdf.cell(width, 7, f"{val:.1f}", border=1, align="C")
    pdf.set_text_color(0, 0, 0)


def _inline_score_bar(pdf: FPDF, label: str, score: float):
    """Draw a small inline progress bar with label and score."""
    x = pdf.get_x() + 5
    y = pdf.get_y()

    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(22, 5, f"  {label}:")

    # Background bar
    bar_x = pdf.get_x()
    bar_w = 50
    bar_h = 4
    pdf.set_fill_color(230, 230, 235)
    pdf.rect(bar_x, y + 0.5, bar_w, bar_h, "F")

    # Filled bar
    fill_w = max(0.5, (score / 100) * bar_w)
    if score >= 70:
        pdf.set_fill_color(34, 139, 34)
    elif score >= 40:
        pdf.set_fill_color(255, 165, 0)
    else:
        pdf.set_fill_color(220, 20, 60)
    pdf.rect(bar_x, y + 0.5, fill_w, bar_h, "F")

    # Score text
    pdf.set_x(bar_x + bar_w + 2)
    pdf.set_font("Helvetica", "B", 8)
    if score >= 70:
        pdf.set_text_color(34, 139, 34)
    elif score >= 40:
        pdf.set_text_color(255, 140, 0)
    else:
        pdf.set_text_color(220, 20, 60)
    pdf.cell(15, 5, f"{score:.0f}", ln=True)
    pdf.set_text_color(0, 0, 0)
