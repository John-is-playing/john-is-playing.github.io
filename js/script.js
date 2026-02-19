// 语言数据
const langData = {
    zh: {
        features: '特性',
        quickstart: '快速上手',
        'api-docs': 'API文档',
        'hero-title': 'EveryThing to EveryThing 类型转换兼容层',
        'hero-subtitle': '一站式解决Python中所有类型转换问题，支持标准类型与第三方库类型的无缝互转',
        'quick-start-btn': '快速开始',
        'api-docs-btn': 'API文档',
        'core-features': '核心特性',
        'comprehensive-type-support': '全面的类型支持',
        'comprehensive-type-support-desc': '支持所有Python标准数据类型和主流第三方库类型，包括numpy、pandas、torch、tensorflow等',
        'bidirectional-conversion': '双向转换',
        'bidirectional-conversion-desc': '提供任意类型之间的双向转换能力，无需手动处理中间格式',
        'type-safety': '类型安全',
        'type-safety-desc': '提供明确的错误提示和类型检查，确保转换过程的安全性',
        'high-performance': '高性能',
        'high-performance-desc': '内置缓存机制，避免重复转换，提升性能',
        'seamless-integration': '无缝集成',
        'seamless-integration-desc': '保持与Python内置函数的兼容性，可直接替换使用',
        'intelligent-processing': '智能处理',
        'intelligent-processing-desc': '自动处理不同库之间的转换细节，如GPU张量的设备管理',
        'installation': '安装',
        'basic-usage': '基本使用',
        'third-party-conversion': '第三方库类型转换',
        'general-conversion': '通用转换方法',
        'supported-types': '支持的类型',
        'standard-types': '标准类型',
        'third-party-libraries': '第三方库',
        'contribution-support': '贡献与支持',
        'contribution-message': '如果您喜欢这个项目，欢迎给我们点个Star，或者参与贡献代码！',
        'project-description': 'EveryThing to EveryThing 类型转换兼容层',
        'copyright': '&copy; 2026 everything2everything. 由 John-is-playing 开发。'
    },
    en: {
        features: 'Features',
        quickstart: 'Quick Start',
        'api-docs': 'API Docs',
        'hero-title': 'EveryThing to EveryThing Type Conversion Compatibility Layer',
        'hero-subtitle': 'A one-stop solution for all Python type conversion problems, supporting seamless conversion between standard types and third-party library types',
        'quick-start-btn': 'Get Started',
        'api-docs-btn': 'API Docs',
        'core-features': 'Core Features',
        'comprehensive-type-support': 'Comprehensive Type Support',
        'comprehensive-type-support-desc': 'Supports all Python standard data types and mainstream third-party library types, including numpy, pandas, torch, tensorflow, etc.',
        'bidirectional-conversion': 'Bidirectional Conversion',
        'bidirectional-conversion-desc': 'Provides bidirectional conversion capabilities between any types without manual handling of intermediate formats',
        'type-safety': 'Type Safety',
        'type-safety-desc': 'Provides clear error messages and type checks to ensure the safety of the conversion process',
        'high-performance': 'High Performance',
        'high-performance-desc': 'Built-in caching mechanism to avoid repeated conversions and improve performance',
        'seamless-integration': 'Seamless Integration',
        'seamless-integration-desc': 'Maintains compatibility with Python built-in functions and can be directly replaced',
        'intelligent-processing': 'Intelligent Processing',
        'intelligent-processing-desc': 'Automatically handles conversion details between different libraries, such as device management for GPU tensors',
        'installation': 'Installation',
        'basic-usage': 'Basic Usage',
        'third-party-conversion': 'Third-party Library Type Conversion',
        'general-conversion': 'General Conversion Method',
        'supported-types': 'Supported Types',
        'standard-types': 'Standard Types',
        'third-party-libraries': 'Third-party Libraries',
        'contribution-support': 'Contribution & Support',
        'contribution-message': 'If you like this project, please give us a Star or contribute code!',
        'project-description': 'EveryThing to EveryThing Type Conversion Compatibility Layer',
        'copyright': '&copy; 2026 everything2everything. Developed by John-is-playing.'
    }
};

// 语言切换函数
function switchLanguage(lang) {
    // 更新所有带有data-lang-key属性的元素
    document.querySelectorAll('[data-lang-key]').forEach(el => {
        const key = el.getAttribute('data-lang-key');
        if (langData[lang] && langData[lang][key]) {
            el.innerHTML = langData[lang][key];
        }
    });
    
    // 更新语言按钮状态
    document.querySelectorAll('.lang-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.getElementById(`lang-${lang}`).classList.add('active');
    
    // 保存语言偏好到本地存储
    localStorage.setItem('language', lang);
    
    // 更新页面标题
    if (lang === 'zh') {
        document.title = 'everything2everything - 全能类型转换兼容层';
    } else {
        document.title = 'everything2everything - Universal Type Conversion Compatibility Layer';
    }
    
    // 更新HTML语言属性
    document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
}

// 平滑滚动功能
document.addEventListener('DOMContentLoaded', function() {
    // 加载保存的语言偏好
    const savedLang = localStorage.getItem('language') || 'zh';
    switchLanguage(savedLang);
    
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
    
    // 语言切换事件监听
    document.getElementById('lang-zh').addEventListener('click', function() {
        switchLanguage('zh');
    });
    
    document.getElementById('lang-en').addEventListener('click', function() {
        switchLanguage('en');
    });
});

// 添加复制按钮和语言切换按钮样式
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

/* 语言切换按钮样式 */
.language-switcher {
    display: flex;
    margin-left: 1rem;
    border-radius: 4px;
    overflow: hidden;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.lang-btn {
    padding: 8px 16px;
    border: none;
    background-color: #f8f9fa;
    color: #333;
    cursor: pointer;
    font-size: 14px;
    transition: all 0.3s ease;
}

.lang-btn:hover {
    background-color: #e9ecef;
}

.lang-btn.active {
    background-color: #3a86ff;
    color: white;
}

/* 响应式调整 */
@media (max-width: 768px) {
    .navbar-links {
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    
    .language-switcher {
        margin-left: 0;
        margin-top: 0.5rem;
    }
}
`;
document.head.appendChild(style);