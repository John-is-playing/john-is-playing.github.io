// 平滑滚动功能
document.addEventListener('DOMContentLoaded', function() {
    // 为所有内部链接添加平滑滚动
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 80, // 减去导航栏高度
                    behavior: 'smooth'
                });
            }
        });
    });

    // 代码块高亮
    const codeBlocks = document.querySelectorAll('pre code');
    codeBlocks.forEach(block => {
        // 添加复制按钮
        const codeBlockElement = block.parentElement;
        const copyButton = document.createElement('button');
        copyButton.className = 'copy-button';
        copyButton.innerHTML = '<i class="fas fa-copy"></i>';
        copyButton.title = '复制代码';
        
        codeBlockElement.style.position = 'relative';
        codeBlockElement.appendChild(copyButton);
        
        // 复制功能
        copyButton.addEventListener('click', function() {
            const code = block.textContent;
            navigator.clipboard.writeText(code).then(() => {
                copyButton.innerHTML = '<i class="fas fa-check"></i>';
                setTimeout(() => {
                    copyButton.innerHTML = '<i class="fas fa-copy"></i>';
                }, 2000);
            }).catch(err => {
                console.error('复制失败:', err);
            });
        });
    });

    // 导航栏滚动效果
    const navbar = document.querySelector('.navbar');
    window.addEventListener('scroll', function() {
        if (window.scrollY > 100) {
            navbar.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.15)';
            navbar.style.transition = 'all 0.3s ease';
        } else {
            navbar.style.boxShadow = '0 2px 4px rgba(0, 0, 0, 0.1)';
        }
    });

    // 特性卡片动画
    const featureCards = document.querySelectorAll('.feature-card');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, { threshold: 0.1 });

    featureCards.forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
        observer.observe(card);
    });

    // 快速上手部分动画
    const quickstartSections = document.querySelectorAll('.quickstart-section');
    quickstartSections.forEach(section => {
        section.style.opacity = '0';
        section.style.transform = 'translateY(20px)';
        section.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
        observer.observe(section);
    });
});

// 添加复制按钮样式
const style = document.createElement('style');
style.textContent = `
.copy-button {
    position: absolute;
    top: 10px;
    right: 10px;
    background-color: rgba(255, 255, 255, 0.8);
    border: none;
    border-radius: 4px;
    padding: 6px 10px;
    cursor: pointer;
    font-size: 14px;
    color: #333;
    transition: all 0.3s ease;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.copy-button:hover {
    background-color: #3a86ff;
    color: white;
}

.copy-button:active {
    transform: scale(0.95);
}
`;
document.head.appendChild(style);