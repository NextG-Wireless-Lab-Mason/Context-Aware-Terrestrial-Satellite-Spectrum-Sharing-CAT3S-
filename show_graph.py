# Import the necessary modules
import os


def plot_graph(img_src):

    # get the absolute path to the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # specify the path to save the html file
    save_path = os.path.join(script_dir, "plot_graph.html")

    # Create the HTML content
    html_content = f"""<!DOCTYPE html>
    <html>
    <body>
        {img_src}
    </body>
    </html>"""

    # Create the new HTML file and write the content to it
    with open(save_path, "w") as f:
        f.write(html_content)

