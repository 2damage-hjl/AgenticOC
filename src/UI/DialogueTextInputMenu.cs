using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using StardewValley;
using StardewValley.Menus;
using System;

namespace ValleyTalk.UI
{
    /// <summary>
    /// 模态文本输入菜单（纯 UI）
    /// 不依赖 NPC / DialogueBuilder
    /// 通过事件向外部报告用户行为
    /// </summary>
    public class DialogueTextInputMenu : IClickableMenu
    {
        public event Action<string>? OnSubmit;
        public event Action? OnCancel;
        public event Action<bool>? OnClearRequested;

        private readonly DialogueTextInputBox _inputBox;
        private readonly ClickableTextureComponent _okButton;
        private readonly ClickableTextureComponent _cancelButton;
        private readonly ClickableTextureComponent _clearButton;

        private const int Width = 1200;
        private const int Height = 600;
        private const int Margin = 24;

        public DialogueTextInputMenu(string title)
            : base(
                (Game1.uiViewport.Width - Width) / 2,
                (Game1.uiViewport.Height - Height) / 2,
                Width,
                Height,
                true
            )
        {
            _inputBox = new DialogueTextInputBox
            {
                Position = new Vector2(xPositionOnScreen + Margin * 2, yPositionOnScreen + 120),
                Extent = new Vector2(Width - Margin * 4, 240)
            };

            _inputBox.OnSubmit += box => Submit(box.Text);
            Game1.keyboardDispatcher.Subscriber = _inputBox;

            _okButton = CreateButton(46, Width - Margin - 64);
            _cancelButton = CreateButton(47, Width - Margin * 2 - 128);
            _clearButton = CreateButton(9, Margin * 2);
        }

        private ClickableTextureComponent CreateButton(int tileIndex, int offsetX)
        {
            return new ClickableTextureComponent(
                new Rectangle(xPositionOnScreen + offsetX, yPositionOnScreen + Height - 96, 64, 64),
                Game1.mouseCursors,
                Game1.getSourceRectForStandardTileSheet(Game1.mouseCursors, tileIndex),
                1f
            );
        }

        public override void draw(SpriteBatch b)
        {
            b.Draw(Game1.fadeToBlackRect, Game1.graphics.GraphicsDevice.Viewport.Bounds, Color.Black * 0.4f);
            Game1.drawDialogueBox(xPositionOnScreen, yPositionOnScreen, width, height, false, true);

            _inputBox.Draw(b);
            _okButton.draw(b);
            _cancelButton.draw(b);
            _clearButton.draw(b);

            drawMouse(b);
        }

        public override void receiveLeftClick(int x, int y, bool playSound = true)
        {
            if (_okButton.containsPoint(x, y))
                Submit(_inputBox.Text);
            else if (_cancelButton.containsPoint(x, y))
                Cancel();
            else if (_clearButton.containsPoint(x, y))
            {
                bool clearAll = Game1.oldKBState.IsKeyDown(Keys.LeftShift);
                OnClearRequested?.Invoke(clearAll);
            }
        }

        public override void receiveKeyPress(Keys key)
        {
            if (key == Keys.Escape)
                Cancel();
            else
                _inputBox.RecieveSpecialInput(key);
        }

        private void Submit(string text)
        {
            exitThisMenu();
            OnSubmit?.Invoke(text);
        }

        private void Cancel()
        {
            exitThisMenu();
            OnCancel?.Invoke();
        }
    }
}
