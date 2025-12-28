import gradio as gr
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    AutoProcessor, AutoModelForCausalLM
)
from PIL import Image
import torch
from typing import Union, Optional, Literal
from pathlib import Path
import gc


class ImageCaptioningModel:
    """
    Универсальный класс для работы с разными моделями описания изображений
    """
    
    SUPPORTED_MODELS = {
        "BLIP Base (990M)": {
            "name": "Salesforce/blip-image-captioning-base",
            "type": "blip",
            "size": "990M параметров",
            "description": "Быстрая и качественная модель для описания изображений"
        },
        "BLIP Large (2.7B)": {
            "name": "Salesforce/blip-image-captioning-large",
            "type": "blip",
            "size": "2.7B параметров",
            "description": "Более точная модель, требует больше ресурсов"
        },
        "GIT Base (700M)": {
            "name": "microsoft/git-base",
            "type": "git",
            "size": "700M параметров",
            "description": "Легкая модель от Microsoft"
        },
        "GIT Large (1.5B)": {
            "name": "microsoft/git-large",
            "type": "git",
            "size": "1.5B параметров",
            "description": "Улучшенная версия GIT"
        },
    }
    
    def __init__(self, device: Optional[str] = None):
        """
        Инициализация без загрузки модели
        """
        self.device = device if device else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.current_model_name = None
        self.model = None
        self.processor = None
        self.model_type = None

    
    def load_model(self, model_key: str) -> str:
        """
        Загрузка выбранной модели
        """
        model_info = self.SUPPORTED_MODELS[model_key]
        model_name = model_info["name"]
        
        if self.current_model_name == model_key:
            return f"Модель {model_key} уже загружена"
        
        # Очищаем предыдущую модель
        if self.model is not None:
            del self.model
            del self.processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        try:
            print(f"Загрузка модели {model_name}...")
            
            if model_info["type"] == "blip":
                self.processor = BlipProcessor.from_pretrained(model_name)
                self.model = BlipForConditionalGeneration.from_pretrained(model_name)
            elif model_info["type"] == "git":
                self.processor = AutoProcessor.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            self.model.to(self.device)
            self.current_model_name = model_key
            self.model_type = model_info["type"]
            
        
        except Exception as e:
            return f"Ошибка при загрузке модели: {str(e)}"
    
    def preprocess_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        """
        Предобработка изображения
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    def generate_caption(
        self,
        image: Union[str, Path, Image.Image],
        max_length: int = 50,
        num_beams: int = 4,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> str:
        """
        Генерация описания изображения
        
        Args:
            image: изображение для описания
            max_length: максимальная длина описания
            num_beams: количество лучей для beam search
            temperature: температура для сэмплирования
            top_p: параметр nucleus sampling
            
        Returns:
            текстовое описание изображения
        """
        if self.model is None:
            return "Сначала загрузите модель!"
        
        try:
            # Предобработка изображения
            image = self.preprocess_image(image)
            
            if self.model_type == "blip":
                return self._generate_blip(image, max_length, num_beams, temperature, top_p)
            elif self.model_type == "git":
                return self._generate_git(image, max_length)
            
        except Exception as e:
            return f"Ошибка при генерации описания: {str(e)}"
    
    def _generate_blip(
        self,
        image: Image.Image,
        max_length: int,
        num_beams: int,
        temperature: float,
        top_p: float
    ) -> str:
        """Генерация описания с помощью BLIP"""
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                top_p=top_p,
                early_stopping=True
            )
        
        return self.processor.decode(outputs[0], skip_special_tokens=True)
    
    def _generate_git(self, image: Image.Image, max_length: int) -> str:
        """Генерация описания с помощью GIT"""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values=inputs.pixel_values,
                max_length=max_length
            )
        
        return self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
    
    
    def get_model_info(self) -> dict:
        """
        Получение информации о текущей модели
        """
        if self.model is None:
            return {
                "status": "Модель не загружена",
                "device": self.device
            }
        
        return {
            "model_name": self.current_model_name,
            "device": self.device,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "model_type": self.model_type
        }
    
    @classmethod
    def get_available_models(cls) -> list:
        """Получение списка доступных моделей"""
        return list(cls.SUPPORTED_MODELS.keys())
    
    @classmethod
    def get_model_description(cls, model_key: str) -> str:
        """Получение описания модели"""
        if model_key in cls.SUPPORTED_MODELS:
            info = cls.SUPPORTED_MODELS[model_key]
            return f"**{model_key}**\n\n📊 {info['size']}\n\n{info['description']}"
        return "Модель не найдена"


class ImageCaptioningApp:
    """
    Класс для создания Gradio приложения с выбором модели
    """
    
    def __init__(self, model: ImageCaptioningModel):
        """
        Инициализация приложения
        """
        self.model = model
        self.demo = self._create_interface()
    
    def _create_interface(self) -> gr.Blocks:
        """
        Создание интерфейса Gradio
        """
        with gr.Blocks(title="Описание изображений с выбором модели", theme=gr.themes.Soft()) as demo:
            gr.Markdown(
                """
                # Описание изображений с помощью AI
                ### Выберите модель и начните работу!
                """
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    model_dropdown = gr.Dropdown(
                        choices=self.model.get_available_models(),
                        label="Выберите модель",
                        value=self.model.get_available_models()[0],
                        interactive=True
                    )
                    
                    load_model_btn = gr.Button("Загрузить модель", variant="primary", size="lg")
                    
                    model_status = gr.Textbox(
                        label="Статус модели",
                        value="Выберите и загрузите модель для начала работы",
                        interactive=False,
                        lines=3
                    )
                
                with gr.Column(scale=1):
                    model_description = gr.Markdown(
                        self.model.get_model_description(self.model.get_available_models()[0])
                    )
                    
                    model_info_json = gr.JSON(
                        label="Информация о модели",
                        value=self.model.get_model_info()
                    )
            
            gr.Markdown("---")
            
            with gr.Tab("Классификация изображения"):
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(
                            type="pil",
                            label="Загрузите изображение"
                        )
                        
                        with gr.Accordion("Настройки генерации", open=False):
                            max_length_slider = gr.Slider(
                                minimum=10,
                                maximum=100,
                                value=50,
                                step=5,
                                label="Максимальная длина описания"
                            )
                            
                            num_beams_slider = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=4,
                                step=1,
                                label="Количество лучей (beam search)"
                            )
                            
                            temperature_slider = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                value=1.0,
                                step=0.1,
                                label="Температура"
                            )
                            
                            top_p_slider = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.9,
                                step=0.05,
                                label="Top-p (nucleus sampling)"
                            )
                        
                        submit_btn = gr.Button("Описать изображение", variant="primary", size="lg")
                    
                    with gr.Column():
                        output_text = gr.Textbox(
                            label="Описание",
                            placeholder="Здесь появится описание изображения...",
                            lines=8,
                            show_copy_button=True
                        )
                
                gr.Examples(
                    examples=[
                        ["https://images.unsplash.com/photo-1583511655857-d19b40a7a54e"],
                        ["https://images.unsplash.com/photo-1543466835-00a7907e9de1"],
                        ["https://images.unsplash.com/photo-1506905925346-21bda4d32df4"],
                        ["https://images.unsplash.com/photo-1472214103451-9374bd1c798e"],
                    ],
                    inputs=image_input,
                    label="📸 Примеры изображений"
                )

            
            with gr.Tab("Информация"):
                gr.Markdown(
                    """
                    ## Руководство по использованию
                    
                    ### Выбор модели
                    - **BLIP Base**: Оптимальный баланс скорости и качества
                    - **BLIP Large**: Максимальное качество, требует больше ресурсов
                    - **GIT Base**: Самая легкая модель, быстрая работа
                    - **GIT Large**: Улучшенное качество от Microsoft
                    
                    ### Параметры генерации
                    - **Максимальная длина**: Определяет детальность описания
                    - **Количество лучей**: Больше лучей = лучше качество, но медленнее
                    - **Температура**: Контролирует креативность (выше = более креативно)
                    - **Top-p**: Фильтрация наименее вероятных слов
                    
                    ### Cоветы
                    - Используйте четкие изображения хорошего качества
                    - Экспериментируйте с разными моделями для лучших результатов
                    - Для быстрой работы используйте BLIP Base или GIT Base
                    - Для максимального качества используйте BLIP Large
                    
                    ### Технические характеристики
                    """
                )
                
                models_comparison = gr.Dataframe(
                    value=[
                        ["BLIP Base", "990M", "Средняя", "Высокое", "Salesforce"],
                        ["BLIP Large", "2.7B", "Медленная", "Очень высокое", "Salesforce"],
                        ["GIT Base", "700M", "Быстрая", "Хорошее", "Microsoft"],
                        ["GIT Large", "1.5B", "Средняя", "Очень хорошее", "Microsoft"],
                    ],
                    headers=["Модель", "Размер", "Скорость", "Качество", "Автор"],
                    label="Сравнение моделей"
                )
            
            def update_model_description(model_key):
                return self.model.get_model_description(model_key)
            
            def load_model_handler(model_key):
                status = self.model.load_model(model_key)
                info = self.model.get_model_info()
                return status, info
            
            model_dropdown.change(
                fn=update_model_description,
                inputs=[model_dropdown],
                outputs=[model_description]
            )
            
            load_model_btn.click(
                fn=load_model_handler,
                inputs=[model_dropdown],
                outputs=[model_status, model_info_json]
            )
            
            submit_btn.click(
                fn=self.model.generate_caption,
                inputs=[
                    image_input,
                    max_length_slider,
                    num_beams_slider,
                    temperature_slider,
                    top_p_slider
                ],
                outputs=output_text
            )
        
        return demo
    
    def launch(
        self,
        share: bool = False,
        server_name: str = "0.0.0.0",
        server_port: int = 7850,
        **kwargs
    ):
        """
        Запуск Gradio приложения
        """
        self.demo.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            **kwargs
        )


# Пример использования
if __name__ == "__main__":
    model = ImageCaptioningModel()
    
    app = ImageCaptioningApp(model)
    app.launch(share=False)
 