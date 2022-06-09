import os, glob
from PIL import Image


def generate_daily_summary(output_file_preix='output_pdf_test'):
    # integrate and merge two images into one
    global svg_files
    print ("Combining pngs to pdfs")
    for png_img in glob.glob("*_radar.png"):
        radar = Image.open(png_img)
        svg_file = png_img.strip("_radar.png") + '.png'
        background = Image.open(svg_file)
        background.paste(radar, box=(1900, 10))
        background.save(svg_file)
        combine = svg_file.split("_")
        svg_files = svg_file.strip(combine[0])
    os.system('convert *'+svg_files+' '+output_file_preix+'.pdf && rm *.png')
    os.system('open '+output_file_preix+'.pdf')
    return True


if __name__ == '__main__':
    generate_daily_summary()