/**
 * Gallery — Alpine.js component.
 *
 * Responsive thumbnail grid displaying generated images.
 * Listens for generation-results and generation-finish events from ws.js.
 *
 * Spec: component-spec-gallery.md
 */

function gallery() {
    return {
        images: [],
        viewMode: 'grid',     // grid | list
        selectedImages: [],   // max 2 IDs for comparison
        lightboxOpen: false,
        lightboxIndex: 0,

        init() {
            window.addEventListener('generation-results', (e) => {
                this._addImages(e.detail.images);
            });
            window.addEventListener('generation-finish', (e) => {
                this._addImages(e.detail.images);
            });
        },

        get imageCount() {
            return this.images.length;
        },

        get isEmpty() {
            return this.images.length === 0;
        },

        get isGenerating() {
            return Alpine.store('generation').isGenerating;
        },

        get previewImage() {
            return Alpine.store('generation').previewImage;
        },

        get progressText() {
            return Alpine.store('generation').progressText;
        },

        imageUrl(img) {
            // Images come as absolute file paths — serve via /outputs/
            if (img.url) return img.url;
            // Extract relative path from absolute path
            const path = img.path || img;
            if (typeof path === 'string' && path.includes('/outputs/')) {
                return '/outputs/' + path.split('/outputs/').pop();
            }
            return path;
        },

        seedLabel(img) {
            return img.seed != null ? `s:${img.seed}` : '';
        },

        selectImage(index) {
            const id = index;
            const idx = this.selectedImages.indexOf(id);
            if (idx >= 0) {
                this.selectedImages.splice(idx, 1);
            } else if (this.selectedImages.length < 2) {
                this.selectedImages.push(id);
            }
        },

        isSelected(index) {
            return this.selectedImages.includes(index);
        },

        openLightbox(index) {
            this.lightboxIndex = index;
            this.lightboxOpen = true;
        },

        closeLightbox() {
            this.lightboxOpen = false;
        },

        lightboxPrev() {
            if (this.lightboxIndex > 0) this.lightboxIndex--;
        },

        lightboxNext() {
            if (this.lightboxIndex < this.images.length - 1) this.lightboxIndex++;
        },

        get lightboxImage() {
            return this.images[this.lightboxIndex];
        },

        _addImages(paths) {
            if (!paths || !Array.isArray(paths)) return;

            const existingPaths = new Set(this.images.map(img => img.path));
            const newImages = [];

            for (const p of paths) {
                const path = typeof p === 'string' ? p : String(p);
                if (existingPaths.has(path)) continue;
                existingPaths.add(path);

                newImages.push({
                    id: `img_${Date.now()}_${this.images.length + newImages.length}`,
                    path: path,
                    url: this._pathToUrl(path),
                    seed: null, // TODO: extract from metadata
                    timestamp: Date.now(),
                });
            }

            if (newImages.length > 0) {
                this.images.push(...newImages);

                // Staggered fade-in via GSAP
                this.$nextTick(() => {
                    const grid = this.$refs?.grid;
                    if (grid && window.fwdFadeIn) {
                        const newEls = Array.from(grid.children).slice(-newImages.length);
                        fwdFadeIn(newEls, 'normal', 0.05);
                    }
                });
            }
        },

        _pathToUrl(path) {
            if (typeof path === 'string' && path.includes('/outputs/')) {
                return '/outputs/' + path.split('/outputs/').pop();
            }
            return path;
        },
    };
}
