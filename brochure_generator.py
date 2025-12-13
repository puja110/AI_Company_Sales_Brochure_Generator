import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from scraper import fetch_website_links, fetch_website_contents
import qrcode
from io import BytesIO
import base64
from PIL import Image
import requests
from colorthief import ColorThief
import re
from urllib.parse import urljoin, urlparse


class BrochureGenerator:
    def __init__(self, model="gpt-4o-mini"):
        """
        Initialize the Brochure Generator with OpenAI API
        
        Args:
            model: The OpenAI model to use (default: gpt-4o-mini)
        """
        load_dotenv(override=True)
        api_key = os.getenv('OPENAI_API_KEY')
        
        # Better error messages
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in .env file!\n"
                "Please create a .env file with: OPENAI_API_KEY=your-key-here"
            )
        
        if not api_key.startswith('sk-'):
            raise ValueError(
                f"Invalid API key format. Key should start with 'sk-' but got: {api_key[:10]}...\n"
                "Please check your .env file."
            )
        
        print(f" API Key loaded successfully (length: {len(api_key)})")
        
        self.openai = OpenAI(api_key=api_key)
        self.model = model
        self.link_selection_model = "gpt-4o-mini"
        
        self.link_system_prompt = """
    You are provided with a list of links found on a webpage.
    You are able to decide which of the links would be most relevant to include in a brochure about the company,
    such as links to an About page, or a Company page, or Careers/Jobs pages.
    You should respond in JSON as in this example:

    {
        "links": [
            {"type": "about page", "url": "https://full.url/goes/here/about"},
            {"type": "careers page", "url": "https://another.full.url/careers"}
        ]
    }
    """
        
        self.brochure_system_prompt = """
    You are an expert marketing copywriter and brochure designer creating professional marketing materials.

    Your task is to create a compelling, visually-structured company brochure that would be used by:
    - Sales teams to pitch to clients
    - Investors for funding decisions  
    - Job seekers to learn about the company
    - Partners for collaboration opportunities

    STRUCTURE YOUR BROCHURE WITH THESE SECTIONS:

    ## Executive Summary
    [2-3 compelling sentences that capture the company's essence and unique value proposition]

    ## About [Company Name]
    [Rich description of the company, its mission, vision, and what makes it special]

    ## What We Do
    [Clear explanation of products/services with benefits focus]

    ## Our Solutions
    [Bullet points of key offerings, each with a brief benefit statement]

    ## Who We Serve
    [Target markets, customer types, industries served]

    ## Why Choose Us?
    [Unique selling points, competitive advantages, key differentiators]

    ## By The Numbers
    [If available: statistics, metrics, achievements, milestones - format as bullet points]

    ## Our Customers
    [If available: customer names, case studies, testimonials]

    ## Recognition & Awards
    [If available: awards, certifications, partnerships]

    ## Company Culture
    [If available: values, work environment, team culture]

    ## Career Opportunities
    [If available: why work here, open positions, benefits]

    ## 📞 Get In Touch
    [Contact information and call-to-action]

    WRITING GUIDELINES:
    - Use engaging, benefit-focused language
    - Keep paragraphs concise (2-4 sentences max)
    - Use bullet points for easy scanning
    - Include specific numbers and metrics when available
    - Write in an enthusiastic but professional tone
    - Focus on outcomes and value, not just features
    - Use action-oriented language
    - Make it skimmable with clear headings and structure

    FORMATTING:
    - Use emojis for visual interest in headings
    - Bold important points
    - Use ">" for quote-style callouts when highlighting key information
    - Create clear visual hierarchy with headers
    - Add horizontal rules (---) to separate major sections

    OUTPUT: Return ONLY the brochure content in markdown format. Do not include code blocks or explanations.
    """

    def get_links_user_prompt(self, url):
        """Create a user prompt for selecting relevant links"""
        user_prompt = f"""
        Here is the list of links on the website {url} -
        Please decide which of these are relevant web links for a brochure about the company, 
        respond with the full https URL in JSON format.
        Do not include Terms of Service, Privacy, email links.

        Links (some might be relative links):

        """
        links = fetch_website_links(url)
        user_prompt += "\n".join(links)
        return user_prompt

    def select_relevant_links(self, url):
        """Use AI to select relevant links from a webpage"""
        print(f" Selecting relevant links for {url}...")
        
        try:
            response = self.openai.chat.completions.create(
                model=self.link_selection_model,
                messages=[
                    {"role": "system", "content": self.link_system_prompt},
                    {"role": "user", "content": self.get_links_user_prompt(url)}
                ],
                response_format={"type": "json_object"}
            )
            
            result = response.choices[0].message.content
            links = json.loads(result)
            print(f" Found {len(links.get('links', []))} relevant links")
            return links
        except Exception as e:
            print(f" Error selecting links: {e}")
            return {"links": []}

    def fetch_page_and_all_relevant_links(self, url):
        """Fetch main page content and all relevant linked pages"""
        print(f"📄 Fetching main page content...")
        contents = fetch_website_contents(url)
        relevant_links = self.select_relevant_links(url)
        
        result = f"## Landing Page:\n\n{contents}\n## Relevant Links:\n"
        
        for link in relevant_links.get('links', []):
            try:
                print(f"📄 Fetching {link['type']}: {link['url']}")
                result += f"\n\n### Link: {link['type']}\n"
                result += fetch_website_contents(link["url"])
            except Exception as e:
                print(f" Could not fetch {link['url']}: {e}")
                continue
        
        return result

    def get_brochure_user_prompt(self, company_name, url):
        """Create the user prompt for brochure generation"""
        user_prompt = f"""
        You are looking at a company called: {company_name}
        Here are the contents of its landing page and other relevant pages;
        use this information to build a short brochure of the company in markdown without code blocks.

        """
        user_prompt += self.fetch_page_and_all_relevant_links(url)
        user_prompt = user_prompt[:15_000]
        return user_prompt

    def create_brochure(self, company_name, url):
        """Generate a brochure for the company (non-streaming)"""
        print(f"\n Generating brochure for {company_name}...\n")
        
        try:
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.brochure_system_prompt},
                    {"role": "user", "content": self.get_brochure_user_prompt(company_name, url)}
                ],
            )
            
            result = response.choices[0].message.content
            print("\n Brochure generated successfully!\n")
            return result
        except Exception as e:
            print(f"\n Error generating brochure: {e}\n")
            return None
        
    def generate_brochure(self, company_name, url):
        """
        Generate a brochure for the company (alias for create_brochure)
        
        Args:
            company_name: Name of the company
            url: Company website URL
            
        Returns:
            Brochure content as markdown string
        """
        return self.create_brochure(company_name, url)

    def stream_brochure(self, company_name, url):
        """Generate a brochure with streaming output (typewriter effect)"""
        print(f"\n Generating brochure for {company_name}...\n")
        print("-" * 80)
        
        try:
            stream = self.openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.brochure_system_prompt},
                    {"role": "user", "content": self.get_brochure_user_prompt(company_name, url)}
                ],
                stream=True
            )
            
            full_response = ""
            for chunk in stream:
                content = chunk.choices[0].delta.content or ''
                full_response += content
                print(content, end='', flush=True)
                yield content
            
            print("\n" + "-" * 80)
            print("\n Brochure generated successfully!\n")
            return full_response
            
        except Exception as e:
            print(f"\n Error generating brochure: {e}\n")
            return None

    def save_brochure(self, brochure_content, filename):
        """Save brochure content to a file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(brochure_content)
            print(f"💾 Brochure saved to {filename}")
        except Exception as e:
            print(f"Error saving brochure: {e}")
    
    # ========================================
    # Color Extraction from Website
    # ========================================
    
    def extract_brand_colors(self, url):
        """
        Extract brand colors by analyzing logo and prominent website colors
        
        Args:
            url: Company website URL
            
        Returns:
            Dict with primary, secondary, accent colors
        """
        try:
            print(f"🎨 Extracting brand colors from {url}...")
            from bs4 import BeautifulSoup
            
            # Default colors
            colors = {
                'primary': '#6366f1',
                'secondary': '#ec4899',
                'accent': '#8b5cf6'
            }
            
            # First, try to extract logo
            logo_data = self.extract_company_logo(url)
            
            # If logo found, extract colors from it
            if logo_data and logo_data.startswith('data:image'):
                try:
                    print("   Extracting colors from logo...")
                    # Decode base64 image
                    img_data_base64 = logo_data.split(',')[1]
                    img_bytes = base64.b64decode(img_data_base64)
                    img_file = BytesIO(img_bytes)
                    
                    # Get dominant colors from logo using ColorThief
                    color_thief = ColorThief(img_file)
                    
                    # Get dominant color
                    dominant = color_thief.get_color(quality=1)
                    colors['primary'] = '#{:02x}{:02x}{:02x}'.format(*dominant)
                    print(f"   Primary (from logo): {colors['primary']}")
                    
                    # Get color palette
                    palette = color_thief.get_palette(color_count=5, quality=1)
                    
                    # Filter out grays and get vibrant colors
                    def is_vibrant(rgb):
                        r, g, b = rgb
                        # Not gray (colors should be different)
                        max_diff = max(abs(r-g), abs(g-b), abs(r-b))
                        if max_diff < 20:
                            return False
                        # Not too dark or too light
                        brightness = (r + g + b) / 3
                        if brightness < 30 or brightness > 240:
                            return False
                        return True
                    
                    vibrant_colors = [c for c in palette if is_vibrant(c)]
                    
                    if len(vibrant_colors) >= 2:
                        colors['secondary'] = '#{:02x}{:02x}{:02x}'.format(*vibrant_colors[1])
                        print(f"   Secondary (from logo): {colors['secondary']}")
                    
                    if len(vibrant_colors) >= 3:
                        colors['accent'] = '#{:02x}{:02x}{:02x}'.format(*vibrant_colors[2])
                        print(f"   Accent (from logo): {colors['accent']}")
                    
                    print(f"Colors extracted from logo successfully")
                    return colors
                    
                except Exception as e:
                    print(f"Could not extract colors from logo: {e}")
            
            # Fallback: Extract from website HTML/CSS
            print("   Extracting colors from website CSS...")
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            soup = BeautifulSoup(response.content, 'html.parser')
            
            all_colors = []
            
            # Look in style tags
            style_tags = soup.find_all('style')
            for style in style_tags:
                if style.string:
                    hex_colors = re.findall(r'#([0-9a-fA-F]{6})', style.string)
                    all_colors.extend(['#' + c.upper() for c in hex_colors])
            
            # Look in inline styles (focus on prominent elements)
            prominent_elements = soup.find_all(['header', 'nav', 'button', 'a'], limit=50)
            for element in prominent_elements:
                if element.get('style'):
                    hex_colors = re.findall(r'#([0-9a-fA-F]{6})', element['style'])
                    all_colors.extend(['#' + c.upper() for c in hex_colors])
            
            # Filter function
            def is_brand_color(color):
                hex_color = color.replace('#', '')
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                
                # Exclude white/light colors
                if r > 240 and g > 240 and b > 240:
                    return False
                
                # Exclude black/dark colors
                if r < 30 and g < 30 and b < 30:
                    return False
                
                # Exclude grays
                max_diff = max(abs(r-g), abs(g-b), abs(r-b))
                if max_diff < 25:
                    return False
                
                return True
            
            # Filter and count
            filtered_colors = [c for c in all_colors if is_brand_color(c)]
            
            if filtered_colors:
                from collections import Counter
                color_counts = Counter(filtered_colors)
                most_common = color_counts.most_common(5)
                unique_colors = [color for color, count in most_common]
                
                if len(unique_colors) >= 1:
                    colors['primary'] = unique_colors[0]
                    print(f"   Primary: {colors['primary']}")
                if len(unique_colors) >= 2:
                    colors['secondary'] = unique_colors[1]
                    print(f"   Secondary: {colors['secondary']}")
                if len(unique_colors) >= 3:
                    colors['accent'] = unique_colors[2]
                    print(f"   Accent: {colors['accent']}")
                
                print(f"Extracted {len(unique_colors)} colors from CSS")
            else:
                print(f"No brand colors found, using defaults")
            
            return colors
            
        except Exception as e:
            print(f"Error: {e}")
            return {
                'primary': '#6366f1',
                'secondary': '#ec4899',
                'accent': '#8b5cf6'
            }
        
    def extract_company_logo(self, url):
        """
        Extract company logo from website
        
        Args:
            url: Company website URL
            
        Returns:
            Base64 encoded logo image or None
        """
        try:
            print(f"🖼️  Extracting logo from {url}...")
            from bs4 import BeautifulSoup
            
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            soup = BeautifulSoup(response.content, 'html.parser')
            
            logo_url = None
            
            # Method 1: Look for meta tags (og:image)
            meta_image = soup.find('meta', property='og:image')
            if meta_image and meta_image.get('content'):
                logo_url = meta_image['content']
                print(f"   Found logo in meta tag")
            
            # Method 2: Look for common logo selectors
            if not logo_url:
                logo_selectors = [
                    'img[class*="logo" i]',
                    'img[id*="logo" i]',
                    'img[alt*="logo" i]',
                    '.logo img',
                    '#logo img',
                    'header img',
                    '.header img',
                    'nav img',
                    '.navbar img',
                    '.navbar-brand img',
                    'a[class*="logo" i] img',
                    '[class*="brand" i] img'
                ]
                
                for selector in logo_selectors:
                    logo_img = soup.select_one(selector)
                    if logo_img and logo_img.get('src'):
                        logo_url = logo_img['src']
                        print(f"   Found logo with selector: {selector}")
                        break
            
            # Method 3: Favicon as fallback
            if not logo_url:
                favicon = soup.find('link', rel='icon') or soup.find('link', rel='shortcut icon')
                if favicon and favicon.get('href'):
                    logo_url = favicon['href']
                    print(f"   Using favicon as logo")
            
            if logo_url:
                # Make URL absolute
                logo_url = urljoin(url, logo_url)
                print(f"   Logo URL: {logo_url}")
                
                # Download and encode image
                img_response = requests.get(logo_url, timeout=10, headers={
                    'User-Agent': 'Mozilla/5.0'
                })
                
                if img_response.status_code == 200:
                    # Convert to base64
                    img_data = base64.b64encode(img_response.content).decode()
                    mime_type = img_response.headers.get('content-type', 'image/png')
                    print(f"Logo extracted successfully")
                    return f"data:{mime_type};base64,{img_data}"
            
            print(f"Could not find logo")
            return None
            
        except Exception as e:
            print(f"Error extracting logo: {e}")
            return None
    
    def extract_company_images(self, url, max_images=6):
        """
        Extract high-quality images from company website
        
        Args:
            url: Company website URL
            max_images: Maximum number of images to extract
            
        Returns:
            List of base64 encoded images
        """
        try:
            print(f"📸 Extracting images from {url}...")
            from bs4 import BeautifulSoup
            
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            soup = BeautifulSoup(response.content, 'html.parser')
            
            images = []
            img_tags = soup.find_all('img')
            
            print(f"   Found {len(img_tags)} image tags")
            
            for img in img_tags:
                if len(images) >= max_images:
                    break
                
                # Get image URL from various attributes
                img_url = img.get('src') or img.get('data-src') or img.get('data-lazy-src') or img.get('data-original')
                if not img_url:
                    continue
                
                # Skip data URIs, SVGs, and other non-image URLs
                if img_url.startswith('data:') or img_url.lower().endswith(('.svg', '.gif')):
                    continue
                
                # Skip small images by checking attributes
                width = img.get('width')
                height = img.get('height')
                
                if width and height:
                    try:
                        w = int(str(width).replace('px', ''))
                        h = int(str(height).replace('px', ''))
                        if w < 200 or h < 150:
                            continue
                    except:
                        pass
                
                # Skip common icon/logo patterns in URL or alt text
                skip_patterns = ['icon', 'logo', 'favicon', 'sprite', 'avatar', 'thumb', 'button', 'badge', 'arrow', 'star', 'check']
                
                # Convert class to string properly
                img_class = img.get('class', [])
                if isinstance(img_class, list):
                    img_class = ' '.join(img_class)
                
                img_str = (img_url + ' ' + str(img.get('alt', '')) + ' ' + str(img_class)).lower()
                if any(pattern in img_str for pattern in skip_patterns):
                    continue
                
                try:
                    # Make URL absolute
                    img_url = urljoin(url, img_url)
                    
                    # Validate URL
                    if not img_url.startswith(('http://', 'https://')):
                        continue
                    
                    # Download image with timeout
                    img_response = requests.get(img_url, timeout=10, headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                        'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8'
                    })
                    
                    # Check if download was successful
                    if img_response.status_code != 200:
                        continue
                    
                    # Check file size (must be at least 10KB, max 5MB)
                    content_length = len(img_response.content)
                    if content_length < 10000 or content_length > 5000000:
                        continue
                    
                    # Try to open and validate the image
                    img_data = BytesIO(img_response.content)
                    
                    try:
                        with Image.open(img_data) as pil_img:
                            # Verify image
                            pil_img.verify()
                    except Exception:
                        # If verify fails, skip this image
                        continue
                    
                    # Reopen image for processing (verify closes the file)
                    img_data = BytesIO(img_response.content)
                    with Image.open(img_data) as pil_img:
                        # Check actual dimensions
                        if pil_img.width < 300 or pil_img.height < 200:
                            continue
                        
                        # Skip if image is mostly transparent or white
                        if pil_img.mode in ('RGBA', 'LA'):
                            # Check alpha channel
                            alpha = pil_img.split()[-1]
                            alpha_data = list(alpha.getdata())
                            # If more than 50% is transparent, skip
                            transparent_pixels = sum(1 for a in alpha_data if a < 128)
                            if transparent_pixels > len(alpha_data) * 0.5:
                                continue
                        
                        # Resize if too large
                        if pil_img.width > 800 or pil_img.height > 600:
                            pil_img.thumbnail((800, 600), Image.Resampling.LANCZOS)
                        
                        # Convert to RGB if needed
                        if pil_img.mode in ('RGBA', 'LA', 'P', 'L'):
                            # Create white background
                            background = Image.new('RGB', pil_img.size, (255, 255, 255))
                            
                            if pil_img.mode == 'P':
                                pil_img = pil_img.convert('RGBA')
                            
                            if pil_img.mode in ('RGBA', 'LA'):
                                # Paste with alpha channel as mask
                                background.paste(pil_img, mask=pil_img.split()[-1])
                            else:
                                background.paste(pil_img)
                            
                            pil_img = background
                        elif pil_img.mode != 'RGB':
                            pil_img = pil_img.convert('RGB')
                        
                        # Check if image is mostly white (boring image)
                        # Sample pixels to check average color
                        pil_img_small = pil_img.resize((50, 50), Image.Resampling.LANCZOS)
                        pixels = list(pil_img_small.getdata())
                        avg_color = [sum(x)/len(pixels) for x in zip(*pixels)]
                        
                        # Skip if too white (all channels > 240)
                        if all(c > 240 for c in avg_color):
                            continue
                        
                        # Convert to base64
                        buffered = BytesIO()
                        pil_img.save(buffered, format="JPEG", quality=85, optimize=True)
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        
                        images.append(f"data:image/jpeg;base64,{img_str}")
                        print(f"Image {len(images)} extracted ({pil_img.width}x{pil_img.height})")
                
                except Exception as e:
                    # Silent fail for individual images
                    continue
            
            print(f"Successfully extracted {len(images)} valid images")
            return images
            
        except Exception as e:
            print(f" Error extracting images: {e}")
            import traceback
            traceback.print_exc()
            return []
        
    def generate_company_headline(self, company_name, url):
        """
        Generate a compelling headline for the company using AI
        
        Args:
            company_name: Name of the company
            url: Company website URL
            
        Returns:
            String with company headline
        """
        try:
            print(f"Generating headline for {company_name}...")
            
            # Fetch a bit of content from the website
            contents = fetch_website_contents(url)
            snippet = contents[:1000]  # First 1000 chars
            
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a marketing copywriter. Create a short, compelling headline (5-10 words) that captures what the company does. Be specific and engaging. Return only the headline, nothing else."},
                    {"role": "user", "content": f"Company: {company_name}\n\nWebsite content: {snippet}\n\nCreate a compelling headline:"}
                ],
                max_tokens=50,
                temperature=0.7
            )
            
            headline = response.choices[0].message.content.strip()
            # Remove quotes if present
            headline = headline.strip('"').strip("'")
            print(f"   ✅ Headline: {headline}")
            return headline
            
        except Exception as e:
            print(f"   ⚠️  Could not generate headline: {e}")
            return "Professional Company Brochure"
        
    def generate_qr_code(self, data, size=200):
        """
        Generate QR code for contact information or URL
        
        Args:
            data: String data to encode (URL, vCard, etc.)
            size: Size of QR code in pixels
            
        Returns:
            Base64 encoded image string
        """
        try:
            import qrcode
            from io import BytesIO
            import base64
            from PIL import Image
            
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_H,
                box_size=10,
                border=4,
            )
            qr.add_data(data)
            qr.make(fit=True)
            
            img = qr.make_image(fill_color="#6366f1", back_color="white")
            img = img.resize((size, size), Image.Resampling.LANCZOS)
            
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return f"data:image/png;base64,{img_str}"
        except Exception as e:
            print(f"Error generating QR code: {e}")
            return None
            
    # ========================================
    # Generate Interactive HTML
    # ========================================
    
    def generate_interactive_html(self, brochure_content, company_name, company_url="", 
                                  animation_style="fade", template_style="professional"):
        """
        Generate an interactive, professionally designed HTML brochure
        """
        import markdown
        
        print(f"Generating interactive HTML brochure for {company_name}...")
        
        # Generate company headline
        company_headline = self.generate_company_headline(company_name, company_url) if company_url else "Professional Company Brochure"
        
        # Extract brand colors (this will also try to get the logo)
        brand_colors = self.extract_brand_colors(company_url) if company_url else {
            'primary': '#6366f1',
            'secondary': '#ec4899',
            'accent': '#8b5cf6'
        }
        
        # Extract logo
        logo_data = self.extract_company_logo(company_url) if company_url else None
        
        # Extract images
        images = self.extract_company_images(company_url, max_images=6) if company_url else []
        
        # Generate QR code for the company URL
        qr_code_data = None
        if company_url:
            print(f"📱 Generating QR code for {company_url}...")
            qr_code_data = self.generate_qr_code(company_url)
            print(f"QR code generated")

        # Convert markdown to HTML
        html_content = markdown.markdown(
            brochure_content,
            extensions=['extra', 'codehilite', 'nl2br', 'tables']
        )
        
        # Logo HTML
        if logo_data:
            logo_html = f'<div class="company-logo"><img src="{logo_data}" alt="{company_name} Logo"></div>'
        else:
            logo_html = f'<div class="company-logo-fallback">{company_name[0].upper() if company_name else "C"}</div>'
        
        # Image Gallery HTML
        gallery_html = ""
        if images:
            gallery_html = '<div class="image-gallery">'
            for i, img in enumerate(images):
                gallery_html += f'''
                <div class="gallery-item">
                    <img src="{img}" alt="Company Image {i+1}">
                </div>
                '''
            gallery_html += '</div>'
        
        # QR Code Section HTML
        qr_section = ""
        if qr_code_data:
            qr_section = f'''
            <div class="qr-section">
                <div class="qr-container">
                    <div class="qr-code">
                        <img src="{qr_code_data}" alt="QR Code">
                    </div>
                    <div class="qr-text">
                        <h3>📱 Quick Access</h3>
                        <p>Scan to visit our website</p>
                        <p class="qr-url">{company_url}</p>
                    </div>
                </div>
            </div>
            '''

        # Generate complete HTML
        return f"""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{company_name} - {company_headline}</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            :root {{
                --primary: {brand_colors['primary']};
                --secondary: {brand_colors['secondary']};
                --accent: {brand_colors['accent']};
                --dark: #1e293b;
                --gray: #64748b;
                --light: #f1f5f9;
                --white: #ffffff;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: var(--dark);
                background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
                padding: 0;
                margin: 0;
            }}
            
            .page-wrapper {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 30px 20px;
            }}
            
            .brochure-header {{
                background: white;
                padding: 40px 40px 35px;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                text-align: center;
                margin-bottom: 30px;
                position: relative;
                overflow: hidden;
            }}
            
            .brochure-header::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 5px;
                background: linear-gradient(90deg, var(--primary), var(--secondary), var(--accent));
            }}
            
            .company-logo {{
                width: 120px;
                height: 120px;
                margin: 0 auto 15px;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 50%;
                overflow: hidden;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                background: white;
                padding: 10px;
            }}
            
            .company-logo img {{
                max-width: 100%;
                max-height: 100%;
                object-fit: contain;
            }}
            
            .company-logo-fallback {{
                width: 120px;
                height: 120px;
                margin: 0 auto 15px;
                background: linear-gradient(135deg, var(--primary), var(--accent));
                color: white;
                font-size: 48px;
                font-weight: bold;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 50%;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }}
            
            .brochure-header h1 {{
                font-size: 2.5rem;
                color: var(--dark);
                margin-bottom: 10px;
                font-weight: 800;
                line-height: 1.2;
            }}
            
            .brochure-subtitle {{
                font-size: 1.2rem;
                color: var(--dark);
                margin-bottom: 15px;
                font-weight: 600;
                line-height: 1.4;
            }}
            
            .header-badge {{
                display: inline-block;
                background: linear-gradient(135deg, var(--primary), var(--accent));
                color: white;
                padding: 8px 20px;
                border-radius: 50px;
                font-weight: 600;
                font-size: 0.85rem;
                margin: 5px 5px 0;
            }}
            
            .image-gallery {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            
            .gallery-item {{
                border-radius: 15px;
                overflow: hidden;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                transition: transform 0.3s ease;
                background: white;
            }}
            
            .gallery-item:hover {{
                transform: translateY(-5px) scale(1.02);
            }}
            
            .gallery-item img {{
                width: 100%;
                height: 220px;
                object-fit: cover;
                display: block;
            }}
            
            .gallery-item img[src=""],
            .gallery-item img:not([src]) {{
                display: none;
            }}
            
            .brochure-content {{
                background: white;
                padding: 45px 45px 40px;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                margin-bottom: 30px;
                color: var(--dark);
            }}
            
            .brochure-content h2 {{
                color: var(--dark);
                font-size: 1.9rem;
                margin-top: 35px;
                margin-bottom: 18px;
                padding-bottom: 12px;
                border-bottom: 3px solid var(--primary);
                position: relative;
                font-weight: 800;
                line-height: 1.3;
            }}
            
            .brochure-content h2:first-child {{
                margin-top: 0;
            }}
            
            .brochure-content h2::after {{
                content: '';
                position: absolute;
                bottom: -3px;
                left: 0;
                width: 80px;
                height: 3px;
                background: var(--secondary);
            }}
            
            .brochure-content h3 {{
                color: var(--dark);
                font-size: 1.4rem;
                margin-top: 25px;
                margin-bottom: 12px;
                font-weight: 700;
                line-height: 1.3;
            }}
            
            .brochure-content h4 {{
                color: var(--dark);
                font-size: 1.2rem;
                margin-top: 20px;
                margin-bottom: 10px;
                font-weight: 600;
                line-height: 1.3;
            }}
            
            .brochure-content h5 {{
                color: var(--dark);
                font-size: 1.05rem;
                margin-top: 16px;
                margin-bottom: 8px;
                font-weight: 600;
                line-height: 1.3;
            }}
            
            .brochure-content h6 {{
                color: var(--dark);
                font-size: 0.95rem;
                margin-top: 14px;
                margin-bottom: 8px;
                font-weight: 600;
                line-height: 1.3;
            }}
            
            .brochure-content p {{
                margin-bottom: 14px;
                font-size: 1.05rem;
                line-height: 1.65;
                color: var(--dark);
            }}
            
            .brochure-content ul,
            .brochure-content ol {{
                margin-left: 28px;
                margin-bottom: 18px;
            }}
            
            .brochure-content li {{
                margin-bottom: 10px;
                font-size: 1rem;
                line-height: 1.6;
                color: var(--dark);
            }}
            
            .brochure-content li::marker {{
                color: var(--primary);
                font-weight: bold;
            }}
            
            .brochure-content blockquote {{
                background: var(--light);
                border-left: 4px solid var(--primary);
                padding: 18px 25px;
                margin: 20px 0;
                border-radius: 8px;
                font-style: italic;
                font-size: 1.05rem;
                color: var(--dark);
                line-height: 1.6;
            }}
            
            .brochure-content strong {{
                color: var(--primary);
                font-weight: 700;
            }}
            
            .brochure-content a {{
                color: var(--primary);
                text-decoration: none;
                border-bottom: 2px solid var(--primary);
                transition: all 0.3s ease;
            }}
            
            .brochure-content a:hover {{
                color: var(--secondary);
                border-bottom-color: var(--secondary);
            }}
            
            .brochure-content hr {{
                border: none;
                height: 2px;
                background: linear-gradient(90deg, var(--primary), var(--secondary), var(--accent));
                margin: 35px 0;
            }}

            /* QR Code Section */
            .qr-section {{
                background: white;
                padding: 40px;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                margin-bottom: 30px;
            }}
            
            .qr-container {{
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 40px;
                flex-wrap: wrap;
            }}
            
            .qr-code {{
                background: white;
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                border: 3px solid var(--light);
            }}
            
            .qr-code img {{
                width: 200px;
                height: 200px;
                display: block;
            }}
            
            .qr-text {{
                text-align: left;
                max-width: 400px;
            }}
            
            .qr-text h3 {{
                color: var(--primary);
                font-size: 1.8rem;
                margin-bottom: 15px;
                font-weight: 800;
            }}
            
            .qr-text p {{
                color: var(--gray);
                font-size: 1.1rem;
                margin-bottom: 10px;
                line-height: 1.6;
            }}
            
            .qr-url {{
                color: var(--primary);
                font-weight: 600;
                word-break: break-all;
            }}
            
            .brochure-footer {{
                background: white;
                padding: 40px 40px 35px;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                text-align: center;
            }}
            
            .brochure-footer h2 {{
                color: var(--dark);
                margin-bottom: 15px;
                font-weight: 800;
                font-size: 1.8rem;
            }}
            
            .brochure-footer p {{
                color: var(--dark);
                line-height: 1.6;
            }}
            
            .cta-button {{
                display: inline-block;
                background: linear-gradient(135deg, var(--primary), var(--accent));
                color: white;
                padding: 16px 35px;
                border-radius: 50px;
                text-decoration: none;
                font-weight: 700;
                font-size: 1.1rem;
                margin: 15px 8px 0;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                transition: all 0.3s ease;
            }}
            
            .cta-button:hover {{
                transform: translateY(-3px);
                box-shadow: 0 15px 40px rgba(0,0,0,0.4);
            }}
            
            @media print {{
                body {{ 
                    background: white; 
                }}
                .page-wrapper {{ 
                    padding: 0; 
                    max-width: 100%;
                }}
                .brochure-header, .brochure-content, .brochure-footer {{
                    box-shadow: none;
                    page-break-inside: avoid;
                    padding: 30px;
                }}
                .cta-button {{ 
                    display: none; 
                }}
                .image-gallery {{ 
                    page-break-inside: avoid;
                }}
                .brochure-content h2 {{
                    page-break-after: avoid;
                }}
            }}
            
            @media (max-width: 768px) {{
                .page-wrapper {{
                    padding: 20px 15px;
                }}
                .brochure-header {{ 
                    padding: 30px 25px 25px; 
                }}
                .brochure-header h1 {{ 
                    font-size: 1.8rem; 
                }}
                .brochure-subtitle {{
                    font-size: 1.05rem;
                }}
                .brochure-content {{ 
                    padding: 30px 25px 25px; 
                }}
                .brochure-content h2 {{ 
                    font-size: 1.6rem; 
                    margin-top: 25px;
                    margin-bottom: 15px;
                }}
                .brochure-content h3 {{
                    font-size: 1.25rem;
                }}
                .brochure-content p {{
                    font-size: 0.98rem;
                }}
                .image-gallery {{ 
                    grid-template-columns: 1fr;
                    gap: 15px;
                }}
                .gallery-item img {{
                    height: 200px;
                }}
                .company-logo,
                .company-logo-fallback {{
                    width: 100px;
                    height: 100px;
                    font-size: 40px;
                }}
            }}
            
            html {{ 
                scroll-behavior: smooth; 
            }}
        </style>
    </head>
    <body>
        <div class="page-wrapper">
            <header class="brochure-header">
                {logo_html}
                <h1>{company_name}</h1>
                <p class="brochure-subtitle">{company_headline}</p>
                {f'<div class="header-badge">🌐 {company_url}</div>' if company_url else ''}
            </header>
            
            {gallery_html}
            
            <main class="brochure-content">
                {html_content}
            </main>

            {qr_section}
            
            <footer class="brochure-footer">
                <h2>Ready to Connect?</h2>
                <p style="font-size: 1.2rem; color: var(--gray); margin: 20px 0;">
                    Get in touch with us today to discover how we can help you achieve your goals.
                </p>
                {f'<a href="{company_url}" class="cta-button">🌐 Visit Website</a>' if company_url else ''}
                <a href="javascript:window.print()" class="cta-button">📄 Print Brochure</a>
                
                <p style="margin-top: 40px; color: var(--gray); font-size: 0.9rem;">
                    Generated by AI Brochure Generator | © {company_name}
                </p>
            </footer>
        </div>
    </body>
    </html>"""